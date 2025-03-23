import os
import sys
import torch
import argparse
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path
from safetensors.torch import load_file

# Import the architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.architecture import IdentityPreservingFlux

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(checkpoint_path, config=None):
    """
    Load a trained identity preservation model
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth or .safetensors)
        config: Optional configuration dictionary
        
    Returns:
        model: Loaded model
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model parameters
    model_params = {}
    if config:
        model_params = {
            'face_embedding_dim': 512,
            'body_embedding_dim': 1024,
            'num_face_latents': config.get('num_face_latents', 16),
            'num_body_latents': config.get('num_body_latents', 16),
            'num_fused_latents': config.get('num_fused_latents', 32),
            'use_identity_fusion': config.get('use_identity_fusion', True),
            'use_gpu': torch.cuda.is_available(),
            'cache_dir': config.get('cache_dir', './cache'),
        }
    
    # Initialize model
    model = IdentityPreservingFlux(**model_params)
    
    # Load base model
    model.load_base_model()
    
    # Check file extension to determine loading method
    if checkpoint_path.endswith('.safetensors'):
        # Load using safetensors
        checkpoint = load_file(checkpoint_path, device=device)
        
        # Handle different state dict keys based on how it was saved
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
            
        # Get metadata if available
        epoch = checkpoint.get('epoch', 0)
        train_metrics = checkpoint.get('train_metrics', {'loss': 0.0})
        val_metrics = checkpoint.get('val_metrics', {'loss': 0.0})
        
        print(f"Model loaded from {checkpoint_path} (safetensors format)")
        if 'epoch' in checkpoint:
            print(f"Trained for {epoch} epochs")
        if 'train_metrics' in checkpoint and 'val_metrics' in checkpoint:
            train_loss = train_metrics.get('loss', 0.0)
            val_loss = val_metrics.get('loss', 0.0)
            print(f"Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
    else:
        # Load using torch.load for .pth files
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different state dict keys
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        # Print model info if available
        if 'epoch' in checkpoint:
            print(f"Model loaded from {checkpoint_path}")
            print(f"Trained for {checkpoint['epoch']} epochs")
        if 'train_loss' in checkpoint and 'val_loss' in checkpoint:
            print(f"Training loss: {checkpoint['train_loss']:.4f}, Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model

def test_identity_transfer(model, reference_image_path, target_image_path, output_path=None):
    """
    Test identity transfer from reference to target
    
    Args:
        model: Trained identity preservation model
        reference_image_path: Path to reference image
        target_image_path: Path to target image
        output_path: Optional path to save output image
        
    Returns:
        output_image: Generated image with identity from reference
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load images
    reference_image = Image.open(reference_image_path).convert('RGB')
    target_image = Image.open(target_image_path).convert('RGB')
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply transforms
    reference_tensor = transform(reference_image).unsqueeze(0).to(device)
    target_tensor = transform(target_image).unsqueeze(0).to(device)
    
    # Extract identity from reference image
    with torch.no_grad():
        ref_face_emb, ref_body_emb, _, _ = model.extract_identity(reference_tensor)
        
        # Prepare identity tokens
        model.prepare_identity_tokens(ref_face_emb, ref_body_emb)
        
        # Generate output with identity preservation
        output_tensor = model(target_tensor)
    
    # Convert output to image
    output_image = output_tensor.squeeze(0).cpu()
    output_image = output_image * 0.5 + 0.5  # Denormalize
    output_image = output_image.clamp(0, 1).permute(1, 2, 0).numpy()
    output_image = (output_image * 255).astype(np.uint8)
    output_image = Image.fromarray(output_image)
    
    # Save output if path provided
    if output_path:
        output_image.save(output_path)
        print(f"Output saved to {output_path}")
    
    # Visualize
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(np.array(reference_image))
    axs[0].set_title("Reference")
    axs[0].axis('off')
    
    axs[1].imshow(np.array(target_image))
    axs[1].set_title("Target")
    axs[1].axis('off')
    
    axs[2].imshow(np.array(output_image))
    axs[2].set_title("Output")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return output_image

def compute_identity_metrics(model, reference_image_path, target_image_path):
    """
    Compute identity metrics between reference and output
    
    Args:
        model: Trained identity preservation model
        reference_image_path: Path to reference image
        target_image_path: Path to target image
        
    Returns:
        metrics: Dictionary of identity metrics
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load images
    reference_image = Image.open(reference_image_path).convert('RGB')
    target_image = Image.open(target_image_path).convert('RGB')
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply transforms
    reference_tensor = transform(reference_image).unsqueeze(0).to(device)
    target_tensor = transform(target_image).unsqueeze(0).to(device)
    
    # Compute metrics
    metrics = {}
    cosine_sim = torch.nn.CosineSimilarity(dim=2)
    
    with torch.no_grad():
        # Extract identity from reference image
        ref_face_emb, ref_body_emb, _, _ = model.extract_identity(reference_tensor)
        
        # Extract identity from target image
        target_face_emb, target_body_emb, _, _ = model.extract_identity(target_tensor)
        
        # Prepare identity tokens
        model.prepare_identity_tokens(ref_face_emb, ref_body_emb)
        
        # Generate output with identity preservation
        output_tensor = model(target_tensor)
        
        # Extract identity from output
        output_face_emb, output_body_emb, _, _ = model.extract_identity(output_tensor)
        
        # Compute cosine similarity between reference and output
        if ref_face_emb.shape == output_face_emb.shape:
            metrics['face_to_ref_cos_sim'] = cosine_sim(ref_face_emb, output_face_emb).mean().item()
        
        if ref_body_emb.shape == output_body_emb.shape:
            metrics['body_to_ref_cos_sim'] = cosine_sim(ref_body_emb, output_body_emb).mean().item()
        
        # Compute cosine similarity between target and output (content preservation)
        if target_face_emb.shape == output_face_emb.shape:
            metrics['face_to_target_cos_sim'] = cosine_sim(target_face_emb, output_face_emb).mean().item()
        
        if target_body_emb.shape == output_body_emb.shape:
            metrics['body_to_target_cos_sim'] = cosine_sim(target_body_emb, output_body_emb).mean().item()
        
        # Compute L2 distance
        if ref_face_emb.shape == output_face_emb.shape:
            metrics['face_to_ref_l2'] = torch.norm(ref_face_emb - output_face_emb, dim=2).mean().item()
        
        if ref_body_emb.shape == output_body_emb.shape:
            metrics['body_to_ref_l2'] = torch.norm(ref_body_emb - output_body_emb, dim=2).mean().item()
    
    return metrics

def batch_test_from_dataset(model, dataset_dir, bundle_id, char_id, output_dir, num_samples=5):
    """
    Run batch test on multiple images from the dataset
    
    Args:
        model: Trained identity preservation model
        dataset_dir: Root directory of dataset
        bundle_id: Bundle ID (e.g., "0")
        char_id: Character ID (e.g., "20")
        output_dir: Directory to save outputs
        num_samples: Number of samples to test
    """
    # Set up paths
    dataset_path = Path(dataset_dir)
    char_path = dataset_path / bundle_id / char_id
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(char_path) if f.endswith('.jpg') or f.endswith('.png')]
    image_files.sort()
    
    # Skip if not enough images
    if len(image_files) < 2:
        print(f"Not enough images for character {bundle_id}_{char_id}, skipping")
        return
    
    # Use first image as reference
    reference_path = str(char_path / image_files[0])
    
    # Process up to num_samples other images as targets
    for i, img_file in enumerate(image_files[1:num_samples+1]):
        target_path = str(char_path / img_file)
        output_path = os.path.join(output_dir, f"{bundle_id}_{char_id}_{i}.jpg")
        
        print(f"Processing {target_path}...")
        
        # Test identity transfer
        test_identity_transfer(model, reference_path, target_path, output_path)
        
        # Compute metrics
        metrics = compute_identity_metrics(model, reference_path, target_path)
        
        # Print metrics
        print(f"Metrics for {bundle_id}_{char_id}_{i}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Test identity preservation model")
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                      help="Path to configuration file")
    parser.add_argument("--reference", type=str, default=None,
                      help="Path to reference image")
    parser.add_argument("--target", type=str, default=None,
                      help="Path to target image")
    parser.add_argument("--output", type=str, default="output.jpg",
                      help="Path to save output image")
    parser.add_argument("--dataset", type=str, default=None,
                      help="Path to dataset for batch testing")
    parser.add_argument("--bundle", type=str, default="0",
                      help="Bundle ID for batch testing")
    parser.add_argument("--char", type=str, default="20",
                      help="Character ID for batch testing")
    parser.add_argument("--output_dir", type=str, default="./test_outputs",
                      help="Directory to save test outputs")
    parser.add_argument("--num_samples", type=int, default=5,
                      help="Number of samples for batch testing")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Load model
    model = load_model(args.checkpoint, config)
    
    # Test options
    if args.reference and args.target:
        # Single test
        test_identity_transfer(model, args.reference, args.target, args.output)
        metrics = compute_identity_metrics(model, args.reference, args.target)
        
        print("Identity metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    elif args.dataset:
        # Batch test from dataset
        batch_test_from_dataset(
            model,
            args.dataset,
            args.bundle,
            args.char,
            args.output_dir,
            args.num_samples
        )
    
    else:
        print("Please provide reference and target images for testing or specify dataset for batch testing.")

if __name__ == "__main__":
    main() 