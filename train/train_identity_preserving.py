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
from safetensors.torch import save_file, load_file

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
                
                # Check for character data pkl file
                pkl_path = os.path.join(char_path, 'char_data.pkl')
                if not os.path.exists(pkl_path):
                    logger.warning(f"No char_data.pkl found in {char_path}, skipping")
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
                        'pkl_path': pkl_path,
                    })
        
        logger.info(f"Dataset loaded with {len(self.samples)} images from {len(set([s['char_id'] for s in self.samples]))} characters")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset:
        - Each sample consists of a reference image and a target image of the same character
        - The reference image is used to extract identity
        - The target image is used for training with its corresponding label from the pickle file
        """
        sample_info = self.samples[idx]
        char_path = sample_info['char_path']
        img_file = sample_info['img_file']
        char_id = sample_info['char_id']
        reference_file = sample_info['reference_file']
        pkl_path = sample_info['pkl_path']
        
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
        
        # Load character data from pickle file
        try:
            with open(pkl_path, 'rb') as f:
                char_data = pickle.load(f)
            
            # Find the label for this specific image
            target_label = None
            for item in char_data:
                if os.path.basename(item.get('image_path', '')) == img_file:
                    target_label = item.get('prompt', f"A photo of character {char_id}")
                    break
                
            # If we didn't find a specific label, use a default one
            if target_label is None:
                target_label = f"A photo of character {char_id}"
        except Exception as e:
            logger.warning(f"Error loading pickle data for {target_img_path}: {e}")
            target_label = f"A photo of character {char_id}"
        
        return {
            'reference_image': ref_img,
            'target_image': target_img,
            'source_image': ref_img,  # Added for compatibility with training loop
            'char_id': char_id,
            'target_path': target_img_path,
            'reference_path': ref_img_path,
            'target_label': target_label,
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
        
    def forward(self, ref_face_embed, ref_body_embed, output_face_embed, output_body_embed, output_image, target_image):
        """
        Calculate the identity preservation loss.
        
        Args:
            output_image: The generated image with identity transfer
            target_image: The original target image
            ref_face_embed: Face embedding from reference image
            ref_body_embed: Body embedding from reference image
            output_face_embed: Face embedding from generated image
            output_body_embed: Body embedding from generated image
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        
        # Face identity loss - cosine similarity between reference and output embeddings
        face_loss = 0
        if ref_face_embed is not None and output_face_embed is not None:
            # Ensure output_face_embed requires gradients
            if not output_face_embed.requires_grad:
                # Create detached copy
                ref_face_flat = ref_face_embed.view(ref_face_embed.size(0), -1).to(self.device).detach()
                # Create copy with requires_grad=True
                output_face_flat = output_face_embed.view(output_face_embed.size(0), -1).to(self.device)
                # If output_face_flat doesn't require gradients, add it explicitly
                if not output_face_flat.requires_grad:
                    output_face_flat = output_face_flat.detach().clone().requires_grad_(True)
            else:
                # Make sure embeddings have the same shape and are on the same device
                ref_face_flat = ref_face_embed.view(ref_face_embed.size(0), -1).to(self.device)
                output_face_flat = output_face_embed.view(output_face_embed.size(0), -1).to(self.device)
            
            # Cosine similarity (higher is better, so we use 1-sim for loss)
            face_sim = self.cosine_sim(ref_face_flat, output_face_flat)
            face_loss = torch.mean(1 - face_sim)
            loss_dict['face_loss'] = face_loss.item()
        
        # Body identity loss - cosine similarity between reference and output embeddings
        body_loss = 0
        if ref_body_embed is not None and output_body_embed is not None:
            # Ensure output_body_embed requires gradients
            if not output_body_embed.requires_grad:
                # Create detached copy
                ref_body_flat = ref_body_embed.view(ref_body_embed.size(0), -1).to(self.device).detach()
                # Create copy with requires_grad=True
                output_body_flat = output_body_embed.view(output_body_embed.size(0), -1).to(self.device)
                # If output_body_flat doesn't require gradients, add it explicitly
                if not output_body_flat.requires_grad:
                    output_body_flat = output_body_flat.detach().clone().requires_grad_(True)
            else:
                # Make sure embeddings have the same shape and are on the same device
                ref_body_flat = ref_body_embed.view(ref_body_embed.size(0), -1).to(self.device)
                output_body_flat = output_body_embed.view(output_body_embed.size(0), -1).to(self.device)
            
            # Cosine similarity (higher is better, so we use 1-sim for loss)
            body_sim = self.cosine_sim(ref_body_flat, output_body_flat)
            body_loss = torch.mean(1 - body_sim)
            loss_dict['body_loss'] = body_loss.item()
        
        # Content preservation loss - VGG features similarity
        content_loss = 0
        if self.content_weight > 0:
            # Ensure output_image requires gradients
            if not output_image.requires_grad:
                output_image = output_image.detach().clone().requires_grad_(True)
            
            # Convert input images to float32 for VGG
            target_float = target_image.float().detach()  # Target doesn't need gradients
            output_float = output_image.float()  # Output needs gradients
            
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

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp):
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        criterion: Loss criterion
        optimizer: Optimizer
        device: Device to use for training
        scaler: GradScaler for AMP
        use_amp: Whether to use AMP
        
    Returns:
        dict: Dictionary with training metrics
    """
    model.train()
    
    # If model has custom training mode method, use it
    if hasattr(model, 'set_training_mode'):
        model.set_training_mode(True)
    
    total_loss = 0
    face_loss_total = 0
    body_loss_total = 0
    content_loss_total = 0
    
    pbar = tqdm(dataloader, desc=f"Training")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Get data
            ref_image = batch['reference_image'].to(device)
            target_image = batch['target_image'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with AMP
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                # Process reference image first to extract identity information
                # This can be done without tracking gradients
                with torch.no_grad():
                    ref_face_embed, ref_body_embed, _, _ = model.extract_identity(ref_image)
                
                # Forward through model to get output image
                output = model(target_image)
                
                # Extract output based on model's return value
                if isinstance(output, tuple):
                    # Handle case where model returns multiple outputs
                    output_image = output[0]
                else:
                    # Model returns just the modified image
                    output_image = output
                
                # For the loss computation, we need to get identity embeddings
                # Create a detached copy for extract_identity (which uses numpy() internally),
                # but keep the original for loss computation to preserve the gradient flow
                with torch.no_grad():
                    # Detach before passing to extract_identity to avoid numpy() gradient error
                    output_image_detached = output_image.detach()
                    output_face_embed, output_body_embed, _, _ = model.extract_identity(output_image_detached)
                
                # Compute loss - since the embeddings are detached, the loss will flow back through output_image
                loss, loss_dict = criterion(
                    ref_face_embed=ref_face_embed,
                    ref_body_embed=ref_body_embed,
                    output_face_embed=output_face_embed,
                    output_body_embed=output_body_embed,
                    output_image=output_image,  # This tensor keeps its gradients for the content loss
                    target_image=target_image
                )
            
            # Backward pass with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            face_loss_total += loss_dict.get('face_loss', 0)
            body_loss_total += loss_dict.get('body_loss', 0)
            content_loss_total += loss_dict.get('content_loss', 0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'face_loss': face_loss_total / (batch_idx + 1),
                'body_loss': body_loss_total / (batch_idx + 1),
                'content_loss': content_loss_total / (batch_idx + 1)
            })
        
        except Exception as e:
            logging.error(f"Error processing batch {batch_idx}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Skip this batch
            continue
    
    # Calculate average metrics
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_face_loss = face_loss_total / num_batches
    avg_body_loss = body_loss_total / num_batches
    avg_content_loss = content_loss_total / num_batches
    
    # Return metrics
    return {
        'loss': avg_loss,
        'face_loss': avg_face_loss,
        'body_loss': avg_body_loss,
        'content_loss': avg_content_loss
    }

def validate(model, dataloader, criterion, device):
    """
    Validate model on validation set
    
    Args:
        model: Model to validate
        dataloader: DataLoader for validation data
        criterion: Loss criterion
        device: Device to use for validation
        
    Returns:
        dict: Dictionary with validation metrics
    """
    model.eval()
    
    # If model has custom training mode method, use it
    if hasattr(model, 'set_training_mode'):
        model.set_training_mode(False)
    
    total_loss = 0
    face_loss_total = 0
    body_loss_total = 0
    content_loss_total = 0
    
    pbar = tqdm(dataloader, desc=f"Validation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                # Get data
                ref_image = batch['reference_image'].to(device)
                target_image = batch['target_image'].to(device)
                
                # Extract identity from reference image
                ref_face_embed, ref_body_embed, _, _ = model.extract_identity(ref_image)
                
                # Forward through model
                output = model(target_image)
                
                # Extract output based on model's return value
                if isinstance(output, tuple):
                    # Handle case where model returns multiple outputs
                    output_image = output[0]
                else:
                    # Model returns just the modified image
                    output_image = output
                
                # Extract identity from output image
                output_face_embed, output_body_embed, _, _ = model.extract_identity(output_image)
                
                # Compute loss - in validation we don't need gradients
                loss, loss_dict = criterion(
                    ref_face_embed=ref_face_embed,
                    ref_body_embed=ref_body_embed,
                    output_face_embed=output_face_embed,
                    output_body_embed=output_body_embed,
                    output_image=output_image,
                    target_image=target_image
                )
                
                # Update metrics
                total_loss += loss.item()
                face_loss_total += loss_dict.get('face_loss', 0)
                body_loss_total += loss_dict.get('body_loss', 0)
                content_loss_total += loss_dict.get('content_loss', 0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'face_loss': face_loss_total / (batch_idx + 1),
                    'body_loss': body_loss_total / (batch_idx + 1),
                    'content_loss': content_loss_total / (batch_idx + 1)
                })
            
            except Exception as e:
                logging.error(f"Error processing validation batch {batch_idx}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                # Skip this batch
                continue
    
    # Calculate average metrics
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_face_loss = face_loss_total / num_batches
    avg_body_loss = body_loss_total / num_batches
    avg_content_loss = content_loss_total / num_batches
    
    # Return metrics
    return {
        'loss': avg_loss,
        'face_loss': avg_face_loss,
        'body_loss': avg_body_loss,
        'content_loss': avg_content_loss
    }

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

def init_model(input_size=512, device='cuda'):
    """
    Initialize the Flux model with identity preservation capabilities
    
    Args:
        input_size: Input image size (assumed square)
        device: Device to put model on ('cuda' or 'cpu')
        
    Returns:
        model: Instance of IdentityPreservingFlux
    """
    try:
        # Create model instance on the specified device
        model = IdentityPreservingFlux(
            use_gpu=device=='cuda',
            cache_dir="./cache"
        )
        model = model.to(device)
        
        # Load base model
        model.load_base_model()
        
        # Log model creation
        logging.info(f"Created IdentityPreservingFlux with input size {input_size} on device {device}")
        
        return model
    except Exception as e:
        logging.error(f"Error initializing IdentityPreservingFlux model: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

def custom_collate_fn(batch):
    """
    Custom collate function to handle PIL images and convert them to tensors
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Collated batch with tensors instead of PIL images
    """
    # Process each element in the batch
    processed_batch = []
    for item in batch:
        processed_item = {}
        for key, value in item.items():
            # Convert PIL images to tensors
            if isinstance(value, Image.Image):
                # Apply transforms to convert PIL image to tensor
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor()
                ])
                value = transform(value)
            processed_item[key] = value
        processed_batch.append(processed_item)
    
    # Now that all PIL images are converted to tensors, use default_collate
    return torch.utils.data.dataloader.default_collate(processed_batch)

def visualize_results(model, dataloader, output_dir, device, epoch, max_samples=8, dpi=150):
    """
    Generate and save visualizations of identity preservation results
        
    Args:
        model: Trained model
        dataloader: DataLoader with test samples
        output_dir: Directory to save visualizations
        device: Device to run inference on
        epoch: Current epoch number
        max_samples: Maximum number of samples to visualize
        dpi: DPI for saved plots
    """
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.set_training_mode(False)
    
    # Get a batch of samples
    batch = next(iter(dataloader))
    source_images = batch['reference_image'].to(device)
    target_images = batch['target_image'].to(device)
    char_ids = batch['char_id']
    
    # Limit samples
    source_images = source_images[:max_samples]
    target_images = target_images[:max_samples]
    char_ids = char_ids[:max_samples]
    
    # Calculate number of rows and columns for grid
    n_samples = source_images.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    
    # Generate output images
    with torch.no_grad():
        # Extract identity from source images
        ref_face_emb, ref_body_emb, _, _ = model.extract_identity(source_images)
        
        # Prepare model with reference identity
        model.prepare_identity_tokens(ref_face_emb, ref_body_emb)
        
        # Generate output images
        output_images = model(target_images)
        
        # Extract identity from output images for similarity measure
        out_face_emb, out_body_emb, _, _ = model.extract_identity(output_images)
    
    # Compute cosine similarity between reference and output embeddings
    cosine_sim = nn.CosineSimilarity(dim=1)
    face_sim = cosine_sim(ref_face_emb.view(ref_face_emb.size(0), -1), 
                         out_face_emb.view(out_face_emb.size(0), -1)).cpu().numpy()
    body_sim = cosine_sim(ref_body_emb.view(ref_body_emb.size(0), -1), 
                         out_body_emb.view(out_body_emb.size(0), -1)).cpu().numpy()
    
    # Convert to numpy for visualization
    source_np = source_images.detach().cpu().permute(0, 2, 3, 1).numpy()
    target_np = target_images.detach().cpu().permute(0, 2, 3, 1).numpy()
    output_np = output_images.detach().cpu().permute(0, 2, 3, 1).numpy()
    
    # Clip values to [0, 1] range
    source_np = np.clip(source_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    output_np = np.clip(output_np, 0, 1)
    
    # Create figure
    fig = plt.figure(figsize=(15, 5 * n_samples))
    
    # Plot images in 3 columns (source, target, output)
    for i in range(n_samples):
        # Source image
        ax1 = fig.add_subplot(n_samples, 3, i*3 + 1)
        ax1.imshow(source_np[i])
        ax1.set_title(f"Source: {char_ids[i]}")
        ax1.axis('off')
        
        # Target image
        ax2 = fig.add_subplot(n_samples, 3, i*3 + 2)
        ax2.imshow(target_np[i])
        ax2.set_title("Target")
        ax2.axis('off')
        
        # Output image
        ax3 = fig.add_subplot(n_samples, 3, i*3 + 3)
        ax3.imshow(output_np[i])
        ax3.set_title(f"Generated (Face Sim: {face_sim[i]:.2f}, Body Sim: {body_sim[i]:.2f})")
        ax3.axis('off')
    
    # Save figure
    plt.tight_layout()
    viz_path = os.path.join(viz_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(viz_path, dpi=dpi)
    
    # Save latest visualization for easy reference
    latest_path = os.path.join(viz_dir, "latest.png")
    plt.savefig(latest_path, dpi=dpi)
    
    plt.close()
    
    # Add extra grid visualization with more samples if available
    if n_samples >= 4:
        # Create a grid figure
        grid_fig = plt.figure(figsize=(15, 15))
        
        # Create three separate grids for source, target, and output
        gs = grid_fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
        
        # Source images grid
        ax_source = grid_fig.add_subplot(gs[0])
        source_grid = torchvision.utils.make_grid(source_images[:16], nrow=4, normalize=True)
        ax_source.imshow(source_grid.permute(1, 2, 0).cpu().numpy())
        ax_source.set_title("Source Images")
        ax_source.axis('off')
        
        # Target images grid
        ax_target = grid_fig.add_subplot(gs[1])
        target_grid = torchvision.utils.make_grid(target_images[:16], nrow=4, normalize=True)
        ax_target.imshow(target_grid.permute(1, 2, 0).cpu().numpy())
        ax_target.set_title("Target Images")
        ax_target.axis('off')
        
        # Output images grid
        ax_output = grid_fig.add_subplot(gs[2])
        output_grid = torchvision.utils.make_grid(output_images[:16], nrow=4, normalize=True)
        ax_output.imshow(output_grid.permute(1, 2, 0).cpu().numpy())
        ax_output.set_title("Generated Images")
        ax_output.axis('off')
        
        # Save grid figure
        plt.tight_layout()
        grid_path = os.path.join(viz_dir, f"grid_epoch_{epoch:03d}.png")
        plt.savefig(grid_path, dpi=dpi)
        
        # Save latest grid for easy reference
        latest_grid_path = os.path.join(viz_dir, "latest_grid.png")
        plt.savefig(latest_grid_path, dpi=dpi)
        
        plt.close()
    
    # Set model back to training mode
    model.set_training_mode(True)
    
    logging.info(f"Saved visualizations to {viz_path}")
    
    return viz_path

def plot_losses(train_losses, val_losses, output_dir, current_epoch, dpi=150):
    """
    Plot training and validation losses and save to output directory
    
    Args:
        train_losses: Dictionary of training losses over epochs
        val_losses: Dictionary of validation losses over epochs
        output_dir: Directory to save plots
        current_epoch: Current epoch number
        dpi: DPI for saved plots
    """
    plt.figure(figsize=(15, 10))
    
    # Create plot directory if it doesn't exist
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot total loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses['total'], label='Train Loss')
    plt.plot(val_losses['total'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot face loss
    plt.subplot(2, 2, 2)
    plt.plot(train_losses['face'], label='Train Face Loss')
    plt.plot(val_losses['face'], label='Val Face Loss')
    plt.title('Face Identity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot body loss
    plt.subplot(2, 2, 3)
    plt.plot(train_losses['body'], label='Train Body Loss')
    plt.plot(val_losses['body'], label='Val Body Loss')
    plt.title('Body Identity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot content loss
    plt.subplot(2, 2, 4)
    plt.plot(train_losses['content'], label='Train Content Loss')
    plt.plot(val_losses['content'], label='Val Content Loss')
    plt.title('Content Preservation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plot_dir, f'losses_epoch_{current_epoch:03d}.png')
    plt.savefig(plot_path, dpi=dpi)
    
    # Save the latest plot too for easy reference
    latest_path = os.path.join(plot_dir, 'latest_losses.png')
    plt.savefig(latest_path, dpi=dpi)
    
    plt.close()
    
    logging.info(f"Saved loss plots to {plot_path}")
    
    return plot_path

def main(config):
    """
    Main training function for identity-preserving model
    
    Args:
        config: Dictionary with training configuration
    """
    try:
        # Initialize logging
        logging_level = getattr(logging, config.get('log_level', 'INFO').upper())
        logging.basicConfig(level=logging_level, 
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Extract parameters from config
        logging.info("Starting identity-preserving model training...")
        
        # Dataset parameters
        data_dir = config.get('dataset_root', '/workspace/dtback/train/dataset')
        # Ensure it's an absolute path
        if not os.path.isabs(data_dir):
            # First try to resolve as a path relative to the current file's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir_candidate = os.path.join(script_dir, data_dir)
            
            # If that doesn't exist, try resolving from the workspace root
            if not os.path.exists(data_dir_candidate):
                workspace_root = os.path.dirname(script_dir)
                data_dir = os.path.join(workspace_root, data_dir)
            else:
                data_dir = data_dir_candidate
            
        # Check if dataset directory exists
        if not os.path.exists(data_dir):
            logging.error(f"Dataset directory not found: {data_dir}")
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        else:
            logging.info(f"Using dataset directory: {data_dir}")
            
        output_dir = config.get('output_dir', 'checkpoints/identity_preserving')
        # Convert to absolute path if relative
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
            
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            logging.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
        # Cache directory for feature extraction
        cache_dir = config.get('cache_dir', 'cache/identity_preserving')
        # Convert to absolute path if relative
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_dir)
            
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            logging.info(f"Creating cache directory: {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)
            
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
        try:
            model = init_model(input_size=input_size, device=device)
            model.to(device)
            logging.info(f"Model created on device: {device}")
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            logging.error("Falling back to default initialization")
            model = IdentityPreservingFlux(
                use_gpu=device=='cuda',
                cache_dir="./cache"
            )
            model.to(device)
        
        # Check if model has custom set_training_mode method
        if hasattr(model, 'set_training_mode'):
            logging.info("Model has custom set_training_mode method - will use this instead of train()")
        
        # Create identity criterion
        criterion = IdentityContentLoss(
            face_weight=face_weight,
            body_weight=body_weight,
            content_weight=content_weight,
            device=device
        ).to(device)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.get('weight_decay', 1e-5))
        
        # Setup AMP scaler
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        # Dynamically scan for bundle directories
        bundle_dirs = None
        if 'bundle_dirs' in config:
            # Use specified bundles if provided in config
            bundle_dirs = config.get('bundle_dirs')
            logging.info(f"Using specified bundle directories: {bundle_dirs}")
        else:
            # Discover available bundles
            bundle_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            logging.info(f"Discovered bundle directories: {bundle_dirs}")
        
        # Create dataset
        try:
            dataset = IdentityDataset(
                root_dir=data_dir,
                bundle_dirs=bundle_dirs,
                max_chars_per_bundle=config.get('max_chars_per_bundle', 100),
                transform=None,  # We'll handle transforms in the dataset class
                use_cache=config.get('use_cache', True),
                cache_dir=cache_dir,
                min_images_per_char=config.get('min_images_per_char', 3),
                reference_image_idx=config.get('reference_image_idx', 0)
            )
            
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )
            
            logging.info(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=config.get('num_workers', 4),
                pin_memory=True,
                collate_fn=custom_collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=config.get('num_workers', 4),
                pin_memory=True,
                collate_fn=custom_collate_fn
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=config.get('num_workers', 4),
                pin_memory=True,
                collate_fn=custom_collate_fn
            )
        except Exception as e:
            logging.error(f"Error creating dataset: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise
        
        # Start training
        logging.info("Starting training...")
        best_val_loss = float('inf')
        
        # Track losses for plotting
        train_losses = {
            'total': [],
            'face': [],
            'body': [],
            'content': []
        }
        
        val_losses = {
            'total': [],
            'face': [],
            'body': [],
            'content': []
        }
        
        # Create visualization directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Visualization frequency
        viz_interval = config.get('viz_interval', 1)  # Visualize every N epochs
        
        for epoch in range(1, num_epochs + 1):
            try:
                logging.info(f"Starting epoch {epoch}/{num_epochs}")
                
                # Train one epoch
                train_metrics = train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                    use_amp=use_amp
                )
                
                # Validate
                val_metrics = validate(
                    model=model,
                    dataloader=val_loader,
                    criterion=criterion,
                    device=device
                )
                
                # Store losses for plotting
                train_losses['total'].append(train_metrics['loss'])
                train_losses['face'].append(train_metrics['face_loss'])
                train_losses['body'].append(train_metrics['body_loss'])
                train_losses['content'].append(train_metrics['content_loss'])
                
                val_losses['total'].append(val_metrics['loss'])
                val_losses['face'].append(val_metrics['face_loss'])
                val_losses['body'].append(val_metrics['body_loss'])
                val_losses['content'].append(val_metrics['content_loss'])
                
                # Log metrics
                logging.info(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                logging.info(f"  Face Loss: Train={train_metrics['face_loss']:.4f}, Val={val_metrics['face_loss']:.4f}")
                logging.info(f"  Body Loss: Train={train_metrics['body_loss']:.4f}, Val={val_metrics['body_loss']:.4f}")
                logging.info(f"  Content Loss: Train={train_metrics['content_loss']:.4f}, Val={val_metrics['content_loss']:.4f}")
                
                # Plot and save losses every epoch
                loss_plot_path = plot_losses(
                    train_losses=train_losses,
                    val_losses=val_losses,
                    output_dir=output_dir,
                    current_epoch=epoch,
                    dpi=config.get('viz_dpi', 150)
                )
                
                # Generate visualizations at specified intervals or if it's the first or last epoch
                if epoch % viz_interval == 0 or epoch == 1 or epoch == num_epochs:
                    try:
                        visualization_path = visualize_results(
                            model=model,
                            dataloader=val_loader,
                            output_dir=output_dir,
                            device=device,
                            epoch=epoch,
                            max_samples=config.get('viz_samples', 8),
                            dpi=config.get('viz_dpi', 150)
                        )
                        logging.info(f"Saved visualization to {visualization_path}")
                    except Exception as e:
                        logging.error(f"Error generating visualizations: {e}")
                        import traceback
                        logging.error(traceback.format_exc())
                
                # Save checkpoint if validation loss improved
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    checkpoint_path = os.path.join(output_dir, f"best_model.safetensors")
                    state_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'face_loss': val_metrics['face_loss'],
                        'body_loss': val_metrics['body_loss'],
                        'content_loss': val_metrics['content_loss'],
                    }
                    # Convert non-tensor values to tensors
                    metadata = {}
                    tensors = {}
                    for k, v in state_dict.items():
                        if isinstance(v, torch.Tensor):
                            tensors[k] = v
                        elif k == 'model_state_dict' or k == 'optimizer_state_dict':
                            for param_key, param_value in v.items():
                                if isinstance(param_value, torch.Tensor):
                                    tensors[f"{k}.{param_key}"] = param_value
                        else:
                            metadata[k] = str(v)
                    
                    save_file(tensors, checkpoint_path, metadata=metadata)
                    logging.info(f"Saved best model checkpoint to {checkpoint_path}")
                    
                    # Also save a copy with epoch number for reference
                    best_epoch_path = os.path.join(output_dir, f"best_model_epoch_{epoch}.safetensors")
                    save_file(tensors, best_epoch_path, metadata=metadata)
                
                # Save regular checkpoint
                if epoch % config.get('save_interval', 10) == 0 or epoch == num_epochs:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.safetensors")
                    state_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'face_loss': val_metrics['face_loss'],
                        'body_loss': val_metrics['body_loss'],
                        'content_loss': val_metrics['content_loss'],
                    }
                    # Convert non-tensor values to tensors
                    metadata = {}
                    tensors = {}
                    for k, v in state_dict.items():
                        if isinstance(v, torch.Tensor):
                            tensors[k] = v
                        elif k == 'model_state_dict' or k == 'optimizer_state_dict':
                            for param_key, param_value in v.items():
                                if isinstance(param_value, torch.Tensor):
                                    tensors[f"{k}.{param_key}"] = param_value
                        else:
                            metadata[k] = str(v)
                    
                    save_file(tensors, checkpoint_path, metadata=metadata)
                    logging.info(f"Saved checkpoint to {checkpoint_path}")
                    
                # Save loss history
                loss_history = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'epochs': list(range(1, epoch + 1))
                }
                loss_history_path = os.path.join(output_dir, 'loss_history.pkl')
                with open(loss_history_path, 'wb') as f:
                    pickle.dump(loss_history, f)
                logging.info(f"Saved loss history to {loss_history_path}")
                    
            except Exception as e:
                logging.error(f"Error in epoch {epoch}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        # Final evaluation on test set
        try:
            test_metrics = validate(
                model=model,
                dataloader=test_loader,
                criterion=criterion,
                device=device
            )
            logging.info(f"Test Loss: {test_metrics['loss']:.4f}")
            logging.info(f"  Face Loss: {test_metrics['face_loss']:.4f}")
            logging.info(f"  Body Loss: {test_metrics['body_loss']:.4f}")
            logging.info(f"  Content Loss: {test_metrics['content_loss']:.4f}")
            
            # Generate final visualizations on test set
            try:
                final_viz_path = visualize_results(
                    model=model,
                    dataloader=test_loader,
                    output_dir=output_dir,
                    device=device,
                    epoch=9999,  # Use a special epoch number to indicate test set
                    max_samples=config.get('viz_samples', 16),  # Show more examples for final evaluation
                    dpi=config.get('viz_dpi', 150)
                )
                logging.info(f"Saved final test set visualization to {final_viz_path}")
            except Exception as e:
                logging.error(f"Error generating final visualizations: {e}")
                import traceback
                logging.error(traceback.format_exc())
            
            # Save final model
            final_path = os.path.join(output_dir, "final_model.safetensors")
            state_dict = {
                'epoch': num_epochs,
                'model_state_dict': model.state_dict(),
                'test_loss': test_metrics['loss'],
                'face_loss': test_metrics['face_loss'],
                'body_loss': test_metrics['body_loss'],
                'content_loss': test_metrics['content_loss'],
            }
            # Convert non-tensor values to tensors
            metadata = {}
            tensors = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    tensors[k] = v
                elif k == 'model_state_dict':
                    for param_key, param_value in v.items():
                        if isinstance(param_value, torch.Tensor):
                            tensors[f"{k}.{param_key}"] = param_value
                else:
                    metadata[k] = str(v)
                    
            save_file(tensors, final_path, metadata=metadata)
            logging.info(f"Saved final model to {final_path}")
            
        except Exception as e:
            logging.error(f"Error in final evaluation: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        return True
        
    except Exception as e:
        logging.error(f"Error in main training function: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train identity preserving model")
    
    # Dataset configuration
    parser.add_argument('--dataset_root', type=str, default='/workspace/dtback/train/dataset',
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
    
    # Visualization parameters
    parser.add_argument('--viz_interval', type=int, default=1,
                        help="Interval for generating visualizations (in epochs)")
    parser.add_argument('--viz_samples', type=int, default=8,
                        help="Number of samples to visualize")
    parser.add_argument('--viz_dpi', type=int, default=150,
                        help="DPI for saved visualizations")
    
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