import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
import warnings
import random
import pickle
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union, Any
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import yaml
import torchvision

# Add parent directory to path to import from train directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from train.architecture import IdentityPreservingFlux
except ImportError:
    try:
        # Try relative import
        from architecture import IdentityPreservingFlux
    except ImportError:
        print("Error: Could not import IdentityPreservingFlux. Make sure the architecture.py file is accessible.")
        sys.exit(1)

try:
    import wandb
except ImportError:
    print("Warning: wandb not installed. Logging will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('identity_training.log'),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger('identity_training')

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

class IdentityDataset(Dataset):
    """Dataset for training identity preservation model"""
    
    def __init__(
        self, 
        root_dir: str, 
        bundle_dirs: Optional[List[str]] = None,
        max_chars_per_bundle: int = 100,
        transform=None, 
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        min_images_per_char: int = 3,
        reference_image_idx: int = 0,
    ):
        """
        Args:
            root_dir: Root directory of the dataset
            bundle_dirs: List of bundle directories to include (if None, use all found)
            max_chars_per_bundle: Maximum number of characters to load per bundle
            transform: Optional transform to apply to images
            use_cache: Whether to cache extracted features
            cache_dir: Directory to save cached features
            min_images_per_char: Minimum number of images required per character
            reference_image_idx: Index of the image to use as reference (typically 0)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_cache = use_cache
        self.min_images_per_char = min_images_per_char
        self.reference_image_idx = reference_image_idx
        
        # Set up cache
        if use_cache:
            self.cache_dir = Path(cache_dir or os.path.join(root_dir, "cache"))
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Find all bundle directories if not specified
        if bundle_dirs is None:
            bundle_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        # Load character data
        self.samples = []
        
        for bundle_dir in bundle_dirs:
            bundle_path = self.root_dir / bundle_dir
            logger.info(f"Loading bundle: {bundle_dir}")
            
            # Find character directories
            char_dirs = [d for d in os.listdir(bundle_path) if os.path.isdir(bundle_path / d)]
            
            # Limit number of characters per bundle if needed
            if max_chars_per_bundle and len(char_dirs) > max_chars_per_bundle:
                random.shuffle(char_dirs)
                char_dirs = char_dirs[:max_chars_per_bundle]
            
            for char_dir in tqdm(char_dirs, desc=f"Loading characters from bundle {bundle_dir}"):
                char_path = bundle_path / char_dir
                
                # Get all image files
                image_files = [f for f in os.listdir(char_path) if f.endswith('.jpg') or f.endswith('.png')]
                
                # Skip characters with insufficient images
                if len(image_files) < self.min_images_per_char:
                    continue
                
                # Sort the images to ensure consistent reference image
                image_files.sort()
                
                # Add all images of this character to our samples list
                for img_file in image_files:
                    self.samples.append({
                        'char_path': str(char_path),
                        'img_file': img_file,
                        'char_id': f"{bundle_dir}_{char_dir}",
                        'is_reference': img_file == image_files[self.reference_image_idx],
                        'reference_file': image_files[self.reference_image_idx],
                    })
        
        logger.info(f"Dataset loaded with {len(self.samples)} images from {len(set([s['char_id'] for s in self.samples]))} characters")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset:
        - Each sample consists of a reference image and a target image of the same character
        - The reference image is used to extract identity
        - The target image is used for training
        """
        sample_info = self.samples[idx]
        char_path = sample_info['char_path']
        img_file = sample_info['img_file']
        char_id = sample_info['char_id']
        reference_file = sample_info['reference_file']
        
        # Load target image
        target_img_path = os.path.join(char_path, img_file)
        target_img = Image.open(target_img_path).convert('RGB')
        
        # Load reference image
        ref_img_path = os.path.join(char_path, reference_file)
        ref_img = Image.open(ref_img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            target_img = self.transform(target_img)
            ref_img = self.transform(ref_img)
        
        return {
            'reference_image': ref_img,
            'target_image': target_img,
            'char_id': char_id,
            'target_path': target_img_path,
            'reference_path': ref_img_path,
        }

class IdentityContentLoss(nn.Module):
    """
    Loss function for identity preservation combining face/body identity and content preservation.
    """
    def __init__(self, face_weight=1.0, body_weight=0.5, content_weight=0.5, device='cuda'):
        super().__init__()
        self.face_weight = face_weight
        self.body_weight = body_weight
        self.content_weight = content_weight
        self.device = device
        
        # Load VGG16 as content model (for perceptual loss)
        vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
        vgg16.eval()
        
        # Extract feature layers for perceptual loss
        self.content_model = nn.Sequential(
            *list(vgg16.features.children())[:16]  # Use features up to relu3_3
        ).to(device)
        
        # Set model to eval mode and disable gradients for content model
        self.content_model.eval()
        for param in self.content_model.parameters():
            param.requires_grad = False
            
        # Convert content model to float32 to avoid mixed precision issues
        self.content_model = self.content_model.float()
            
        # Define MSE loss for content comparison
        self.mse_loss = nn.MSELoss()
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        
    def forward(self, output_image, target_image, ref_face_emb, ref_body_emb, output_face_emb, output_body_emb):
        """
        Calculate the identity preservation loss.
        
        Args:
            output_image: The generated image with identity transfer
            target_image: The original target image
            ref_face_emb: Face embedding from reference image
            ref_body_emb: Body embedding from reference image
            output_face_emb: Face embedding from generated image
            output_body_emb: Body embedding from generated image
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        
        # Face identity loss - cosine similarity between reference and output embeddings
        face_loss = 0
        if ref_face_emb is not None and output_face_emb is not None:
            # Make sure embeddings have the same shape and are on the same device
            ref_face_flat = ref_face_emb.view(ref_face_emb.size(0), -1).to(self.device)
            output_face_flat = output_face_emb.view(output_face_emb.size(0), -1).to(self.device)
            
            # Cosine similarity (higher is better, so we use 1-sim for loss)
            face_sim = self.cosine_sim(ref_face_flat, output_face_flat)
            face_loss = torch.mean(1 - face_sim)
            loss_dict['face_loss'] = face_loss.item()
        
        # Body identity loss - cosine similarity between reference and output embeddings
        body_loss = 0
        if ref_body_emb is not None and output_body_emb is not None:
            # Make sure embeddings have the same shape and are on the same device
            ref_body_flat = ref_body_emb.view(ref_body_emb.size(0), -1).to(self.device)
            output_body_flat = output_body_emb.view(output_body_emb.size(0), -1).to(self.device)
            
            # Cosine similarity (higher is better, so we use 1-sim for loss)
            body_sim = self.cosine_sim(ref_body_flat, output_body_flat)
            body_loss = torch.mean(1 - body_sim)
            loss_dict['body_loss'] = body_loss.item()
        
        # Content preservation loss - VGG features similarity
        content_loss = 0
        if self.content_weight > 0:
            # Convert input images to float32 for VGG
            target_float = target_image.float()
            output_float = output_image.float()
            
            # Extract VGG features
            with torch.no_grad():
                target_features = self.content_model(target_float)
            output_features = self.content_model(output_float)
            
            # Calculate MSE between features
            content_loss = self.mse_loss(output_features, target_features)
            loss_dict['content_loss'] = content_loss.item()
        
        # Combine losses with weights
        total_loss = (
            self.face_weight * face_loss + 
            self.body_weight * body_loss + 
            self.content_weight * content_loss
        )
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False):
    """
    Train the model for one epoch
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for AMP
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        metrics: Dictionary with training metrics
    """
    model.set_training_mode(True)
    losses = []
    
    # Dictionary to track metrics
    metrics = {
        'loss': 0.0,
        'face_loss': 0.0,
        'body_loss': 0.0,
        'content_loss': 0.0
    }
    
    # Track batch count for averaging
    batch_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Extract source and target images
        source_images = batch['source_image'].to(device)
        target_images = batch['target_image'].to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Use AMP if enabled
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Extract reference identity from source images
            ref_identity = model.extract_identity(source_images)
            
            # Generate output images using the model
            output_images, output_identity = model(target_images)
            
            # Compute loss
            loss, loss_dict = criterion(
                output_image=output_images,
                target_image=target_images,
                ref_face_emb=ref_identity,
                ref_body_emb=None,
                output_face_emb=output_identity,
                output_body_emb=None
            )
        
        # Backward pass with AMP scaler if enabled
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update metrics
        for key, value in loss_dict.items():
            if key in metrics:
                metrics[key] += value
        
        # Track losses for logging
        losses.append(loss.item())
        batch_count += 1
        
        # Log progress
        if batch_idx % 10 == 0:
            logging.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    # Compute average metrics
    for key in metrics:
        metrics[key] /= max(batch_count, 1)
    
    return metrics

def validate(model, dataloader, criterion, device):
    """
    Validate the model
    
    Args:
        model: Model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        metrics: Dictionary with validation metrics
    """
    model.set_training_mode(False)
    losses = []
    
    # Dictionary to track metrics
    metrics = {
        'loss': 0.0,
        'face_loss': 0.0,
        'body_loss': 0.0,
        'content_loss': 0.0
    }
    
    # Track batch count for averaging
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Extract source and target images
            source_images = batch['source_image'].to(device)
            target_images = batch['target_image'].to(device)
            
            # Extract reference identity from source images
            ref_identity = model.extract_identity(source_images)
            
            # Generate output images using the model
            output_images, output_identity = model(target_images)
            
            # Compute loss
            loss, loss_dict = criterion(
                output_image=output_images,
                target_image=target_images,
                ref_face_emb=ref_identity,
                ref_body_emb=None,
                output_face_emb=output_identity,
                output_body_emb=None
            )
            
            # Update metrics
            for key, value in loss_dict.items():
                if key in metrics:
                    metrics[key] += value
            
            # Track losses for logging
            losses.append(loss.item())
            batch_count += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logging.info(f"Validation Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    # Compute average metrics
    for key in metrics:
        metrics[key] /= max(batch_count, 1)
    
    return metrics

def visualize_examples(examples, epoch):
    """
    Visualize example outputs and log to wandb
    
    Args:
        examples: List of example outputs
        epoch: Current epoch number
    """
    if not examples:
        logger.warning("No examples provided for visualization")
        return
    
    try:
        # Create a grid of examples
        num_examples = len(examples)
        fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))
        
        if num_examples == 1:
            axes = [axes]
        
        for i, example in enumerate(examples):
            try:
                # Convert tensors to numpy arrays
                ref_img = example['reference'].permute(1, 2, 0).numpy()
                target_img = example['target'].permute(1, 2, 0).numpy()
                output_img = example['output'].permute(1, 2, 0).numpy()
                
                # Ensure all tensors have valid values
                for img, name in [(ref_img, 'reference'), (target_img, 'target'), (output_img, 'output')]:
                    if np.isnan(img).any():
                        logger.warning(f"NaN values detected in {name} image")
                        img = np.nan_to_num(img)
                    if np.isinf(img).any():
                        logger.warning(f"Inf values detected in {name} image")
                        img = np.nan_to_num(img, posinf=1.0, neginf=0.0)
                
                # Normalize images to [0, 1] range with error handling
                for img, name in [(ref_img, 'reference'), (target_img, 'target'), (output_img, 'output')]:
                    min_val = img.min()
                    max_val = img.max()
                    if min_val == max_val:
                        logger.warning(f"{name} image has constant values, skipping normalization")
                    else:
                        img = (img - min_val) / (max_val - min_val)
                        
                # Ensure images are in valid range
                ref_img = np.clip(ref_img, 0, 1)
                target_img = np.clip(target_img, 0, 1)
                output_img = np.clip(output_img, 0, 1)
                
                # Plot images
                axes[i][0].imshow(ref_img)
                axes[i][0].set_title("Reference")
                axes[i][0].axis('off')
                
                axes[i][1].imshow(target_img)
                axes[i][1].set_title("Target")
                axes[i][1].axis('off')
                
                axes[i][2].imshow(output_img)
                axes[i][2].set_title("Output")
                axes[i][2].axis('off')
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                # Continue with the next example
                continue
        
        plt.tight_layout()
        
        # Log to wandb
        if 'wandb' in sys.modules and wandb.run is not None:
            try:
                wandb.log({f"examples/epoch_{epoch}": wandb.Image(fig)})
            except Exception as e:
                logger.error(f"Error logging to wandb: {e}")
        
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error in visualize_examples: {e}")
        import traceback
        logger.error(traceback.format_exc())

def identity_similarity_metrics(model, test_loader, device):
    """
    Compute identity preservation metrics on the test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        metrics: Dictionary of metrics
    """
    model.eval()
    metrics = {
        'face_cos_similarity': [],
        'body_cos_similarity': [],
        'face_l2_distance': [],
        'body_l2_distance': [],
    }
    
    cosine_sim = nn.CosineSimilarity(dim=2)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing metrics"):
            # Get images
            reference_images = batch['reference_image'].to(device)
            target_images = batch['target_image'].to(device)
            
            # Extract identity from reference images
            ref_face_emb, ref_body_emb, _, _ = model.extract_identity(reference_images)
            
            # Prepare identity tokens
            model.prepare_identity_tokens(ref_face_emb, ref_body_emb)
            
            # Generate output with identity preservation
            output = model(target_images)
            
            # Extract identity from output images
            output_face_emb, output_body_emb, _, _ = model.extract_identity(output)
            
            # Extract identity from target images (ground truth)
            target_face_emb, target_body_emb, _, _ = model.extract_identity(target_images)
            
            # Compute cosine similarity
            if output_face_emb.shape == target_face_emb.shape:
                face_cos = cosine_sim(output_face_emb, target_face_emb).mean().item()
                metrics['face_cos_similarity'].append(face_cos)
            
            if output_body_emb.shape == target_body_emb.shape:
                body_cos = cosine_sim(output_body_emb, target_body_emb).mean().item()
                metrics['body_cos_similarity'].append(body_cos)
            
            # Compute L2 distance
            if output_face_emb.shape == target_face_emb.shape:
                face_l2 = torch.norm(output_face_emb - target_face_emb, dim=2).mean().item()
                metrics['face_l2_distance'].append(face_l2)
            
            if output_body_emb.shape == target_body_emb.shape:
                body_l2 = torch.norm(output_body_emb - target_body_emb, dim=2).mean().item()
                metrics['body_l2_distance'].append(body_l2)
    
    # Compute average metrics
    avg_metrics = {}
    for key, values in metrics.items():
        if values:
            avg_metrics[key] = sum(values) / len(values)
        else:
            avg_metrics[key] = float('nan')
    
    return avg_metrics

class SimpleIdentityAutoencoder(nn.Module):
    """A simple identity-preserving autoencoder model"""
    
    def __init__(self, input_size=512, device='cuda'):
        super(SimpleIdentityAutoencoder, self).__init__()
        
        # Set device
        self.device = device
        self.training_mode = True
        
        # Encoder (downsampling path)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1).to(device),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1).to(device),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).to(device)
        
        # Identity embedding layer
        self.identity_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1).to(device),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        ).to(device)
        
        # Decoder (upsampling path)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1).to(device),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1).to(device),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1).to(device),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1).to(device),
            nn.Sigmoid()  # Output in range [0, 1]
        ).to(device)
        
    def set_training_mode(self, mode):
        """Set training mode for the model"""
        self.training_mode = mode
        if mode:
            self.train()
        else:
            self.eval()
        return self
        
    def extract_identity(self, x):
        """Extract identity features from input image"""
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Get encoder features
        features = self.encoder(x)
        
        # Extract identity
        identity = self.identity_layer(features)
        
        # Flatten to vector
        identity = identity.view(identity.size(0), -1)
        
        return identity
        
    def prepare_identity_tokens(self, identity_embeddings, target_shape):
        """
        Prepare identity tokens for conditioning the decoder
        
        Args:
            identity_embeddings: Identity embeddings from extract_identity
            target_shape: Shape to match for decoder input
            
        Returns:
            Reshaped identity tokens
        """
        batch_size, channels, height, width = target_shape
        
        # Reshape identity embeddings to match target shape
        identity_tokens = identity_embeddings.view(batch_size, -1, 1, 1)
        identity_tokens = identity_tokens.expand(-1, -1, height, width)
        
        return identity_tokens
        
    def forward(self, x, identity_embeddings=None):
        """
        Forward pass through the model
        
        Args:
            x: Input image
            identity_embeddings: Optional identity embeddings for reconstruction
            
        Returns:
            output: Reconstructed image with identity preserved
            identity: Identity embeddings
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # 1. Encode input
        encoded = self.encoder(x)
        
        # 2. Extract identity features if needed and if in training mode
        if identity_embeddings is None:
            identity = self.identity_layer(encoded)
            identity = identity.view(identity.size(0), -1)  # Flatten to vector
        else:
            # Use provided identity embeddings
            identity = identity_embeddings
        
        # 3. Decode with identity features to reconstruct image
        # If we have identity embeddings provided, use those instead
        reconstructed = self.decoder(encoded)
        
        return reconstructed, identity

def init_model(input_size=512, device='cuda'):
    """
    Initialize a simple identity-preserving autoencoder model
    
    Args:
        input_size: Input image size (assumed square)
        device: Device to put model on ('cuda' or 'cpu')
        
    Returns:
        model: Instance of SimpleIdentityAutoencoder
    """
    # Create model instance on the specified device
    model = SimpleIdentityAutoencoder(input_size=input_size, device=device)
    model = model.to(device)
    
    # Log model creation
    logging.info(f"Created SimpleIdentityAutoencoder with input size {input_size} on device {device}")
    
    return model

def main(config):
    """
    Main training function for identity-preserving model
    
    Args:
        config: Dictionary with training configuration
    """
    try:
        # Extract parameters from config
        logging.info("Starting identity-preserving model training...")
        
        # Dataset parameters
        data_dir = config.get('data_dir', 'data/identity_images')
        output_dir = config.get('output_dir', 'checkpoints/identity_preserving')
        batch_size = config.get('batch_size', 4)
        
        # Learning parameters
        num_epochs = config.get('num_epochs', 50)
        learning_rate = config.get('learning_rate', 3e-4)
        face_weight = config.get('face_weight', 1.0)
        body_weight = config.get('body_weight', 0.5)
        content_weight = config.get('content_weight', 0.5)
        
        # Optimization parameters
        use_amp = config.get('use_amp', True)
        
        # Model parameters
        input_size = config.get('input_size', 512)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        model = init_model(input_size=input_size, device=device)
        model.to(device)
        logging.info(f"Model created on device: {device}")
        
        # Create identity criterion
        criterion = IdentityContentLoss(
            face_weight=face_weight,
            body_weight=body_weight,
            content_weight=content_weight,
            device=device
        ).to(device)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Setup AMP scaler
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        # Create training and validation datasets
        # This is a simple example - you should replace with your actual dataset
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
        
        # Create dummy datasets for demonstration
        # Replace with your actual dataset implementation
        class DummyIdentityDataset(Dataset):
            def __init__(self, size=100, image_size=512):
                self.size = size
                self.image_size = image_size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Create random source and target images
                source_img = torch.rand(3, self.image_size, self.image_size)
                target_img = torch.rand(3, self.image_size, self.image_size)
                
                # Return source and target
                return {
                    'source_image': source_img,
                    'target_image': target_img
                }
        
        # Create datasets and dataloaders
        train_dataset = DummyIdentityDataset(size=1000, image_size=input_size)
        val_dataset = DummyIdentityDataset(size=100, image_size=input_size)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            logging.info(f"Starting epoch {epoch}/{num_epochs}")
            
            # Set model to training mode
            model.set_training_mode(True)
            
            # Training phase
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp
            )
            
            # Log training metrics
            logging.info(f"Epoch {epoch} training: {train_metrics}")
            
            # Set model to evaluation mode
            model.set_training_mode(False)
            
            # Validation phase
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device
            )
            
            # Log validation metrics
            logging.info(f"Epoch {epoch} validation: {val_metrics}")
            
            # Save checkpoint
            if epoch % 5 == 0 or epoch == num_epochs:
                checkpoint_path = os.path.join(output_dir, f"identity_model_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                }, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save final model
        final_model_path = os.path.join(output_dir, "identity_model_final.pth")
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Saved final model to {final_model_path}")
        
        logging.info("Training completed successfully!")
        return model
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train identity preserving model")
    
    # Dataset configuration
    parser.add_argument('--dataset_root', type=str, default='dataset_creation/data/dataset',
                        help="Root directory of the dataset")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for training")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument('--val_split', type=float, default=0.1,
                        help="Validation split ratio")
    
    # Model configuration
    parser.add_argument('--input_size', type=tuple, default=(512, 512),
                        help="Input image size")
    
    # Loss configuration
    parser.add_argument('--face_weight', type=float, default=1.0,
                        help="Weight for face identity loss")
    parser.add_argument('--body_weight', type=float, default=0.5,
                        help="Weight for body identity loss")
    parser.add_argument('--content_weight', type=float, default=0.5,
                        help="Weight for content preservation loss")
    
    # Training configuration
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument('--num_epochs', type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to train on (cuda or cpu)")
    parser.add_argument('--use_amp', action='store_true',
                        help="Use automatic mixed precision")
    
    # Logging and saving configuration
    parser.add_argument('--output_dir', type=str, default='./checkpoints/identity_preserving',
                        help="Directory to save model checkpoints")
    parser.add_argument('--log_interval', type=int, default=10,
                        help="Interval for logging batch results")
    parser.add_argument('--val_interval', type=int, default=1,
                        help="Interval for validation (in epochs)")
    parser.add_argument('--save_interval', type=int, default=5,
                        help="Interval for saving checkpoints (in epochs)")
    
    # Weights & Biases configuration
    parser.add_argument('--use_wandb', action='store_true',
                        help="Use Weights & Biases for logging")
    parser.add_argument('--wandb_project', type=str, default='identity-preserving-flux',
                        help="Weights & Biases project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="Weights & Biases entity name")
    parser.add_argument('--wandb_name', type=str, default=None,
                        help="Weights & Biases run name")
    
    args = parser.parse_args()
    
    # Convert args to dictionary
    config = vars(args)
    
    # Run main
    main(config) 