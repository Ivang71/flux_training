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
import torch.nn as nn
import logging
import time
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.architecture import IdentityPreservingFlux

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(checkpoint_path, config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    model = IdentityPreservingFlux(**model_params)
    print(f"Model created with config: {model_params}")
    
    model.load_base_model()
    
    if checkpoint_path.endswith('.safetensors'):
        print(f"Loading safetensors checkpoint from {checkpoint_path}")
        loaded_state_dict = load_file(checkpoint_path)
        
        metadata = {}
        for key in loaded_state_dict:
            if not key.startswith('model_state_dict.'):
                try:
                    if loaded_state_dict[key].numel() == 1:
                        value = loaded_state_dict[key].item()
                        metadata[key] = value
                except:
                    pass
        
        model_state_dict = {}
        for key in loaded_state_dict:
            if key.startswith('model_state_dict.'):
                param_name = key[len('model_state_dict.'):]
                model_state_dict[param_name] = loaded_state_dict[key]
        
        model.load_state_dict(model_state_dict)
        
        epoch = metadata.get('epoch', 0)
        train_metrics = metadata.get('train_metrics', {'loss': 0.0})
        val_metrics = metadata.get('val_metrics', {'loss': 0.0})
        
        print(f"Model loaded from {checkpoint_path} (safetensors format)")
        if 'epoch' in metadata:
            print(f"Trained for {epoch} epochs")
        if 'train_metrics' in metadata and 'val_metrics' in metadata:
            train_loss = train_metrics.get('loss', 0.0)
            val_loss = val_metrics.get('loss', 0.0)
            print(f"Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        if 'epoch' in checkpoint:
            print(f"Model loaded from {checkpoint_path}")
            print(f"Trained for {checkpoint['epoch']} epochs")
        if 'train_loss' in checkpoint and 'val_loss' in checkpoint:
            print(f"Training loss: {checkpoint['train_loss']:.4f}, Validation loss: {checkpoint['val_loss']:.4f}")
    
    model = model.to(device)
    model.eval()
    
    return model

def test_identity_transfer(model, reference_image_path, target_image_path, output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    reference_image = Image.open(reference_image_path).convert('RGB')
    target_image = Image.open(target_image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    reference_tensor = transform(reference_image).unsqueeze(0).to(device)
    target_tensor = transform(target_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ref_face_emb, ref_body_emb, _, _ = model.extract_identity(reference_tensor)
        
        model.prepare_identity_tokens(ref_face_emb, ref_body_emb)
        
        output_tensor = model(target_tensor)
    
    output_image = output_tensor.squeeze(0).cpu()
    output_image = output_image * 0.5 + 0.5
    output_image = output_image.clamp(0, 1).permute(1, 2, 0).numpy()
    output_image = (output_image * 255).astype(np.uint8)
    output_image = Image.fromarray(output_image)
    
    if output_path:
        output_image.save(output_path)
        print(f"Output saved to {output_path}")
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    reference_image = Image.open(reference_image_path).convert('RGB')
    target_image = Image.open(target_image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    reference_tensor = transform(reference_image).unsqueeze(0).to(device)
    target_tensor = transform(target_image).unsqueeze(0).to(device)
    
    metrics = {}
    cosine_sim = torch.nn.CosineSimilarity(dim=2)
    
    with torch.no_grad():
        ref_face_emb, ref_body_emb, _, _ = model.extract_identity(reference_tensor)
        
        target_face_emb, target_body_emb, _, _ = model.extract_identity(target_tensor)
        
        model.prepare_identity_tokens(ref_face_emb, ref_body_emb)
        
        output_tensor = model(target_tensor)
        
        output_face_emb, output_body_emb, _, _ = model.extract_identity(output_tensor)
        
        if ref_face_emb is not None and output_face_emb is not None:
            face_sim = cosine_sim(ref_face_emb.view(ref_face_emb.size(0), 1, -1), 
                                 output_face_emb.view(output_face_emb.size(0), 1, -1)).mean().item()
            metrics['face_similarity'] = face_sim
        
        if ref_body_emb is not None and output_body_emb is not None:
            body_sim = cosine_sim(ref_body_emb.view(ref_body_emb.size(0), 1, -1),
                                 output_body_emb.view(output_body_emb.size(0), 1, -1)).mean().item()
            metrics['body_similarity'] = body_sim
        
        if target_face_emb is not None and output_face_emb is not None:
            target_face_sim = cosine_sim(target_face_emb.view(target_face_emb.size(0), 1, -1),
                                        output_face_emb.view(output_face_emb.size(0), 1, -1)).mean().item()
            metrics['target_face_similarity'] = target_face_sim
        
        if target_body_emb is not None and output_body_emb is not None:
            target_body_sim = cosine_sim(target_body_emb.view(target_body_emb.size(0), 1, -1),
                                        output_body_emb.view(output_body_emb.size(0), 1, -1)).mean().item()
            metrics['target_body_similarity'] = target_body_sim
    
    print("Identity Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return metrics

def batch_test_from_dataset(model, dataset_dir, bundle_id, char_id, output_dir, num_samples=5):
    bundle_path = os.path.join(dataset_dir, bundle_id)
    char_path = os.path.join(bundle_path, char_id)
    
    if not os.path.exists(char_path):
        print(f"Character path not found: {char_path}")
        return
    
    image_files = [f for f in os.listdir(char_path) if f.endswith('.jpg') or f.endswith('.png')]
    image_files.sort()
    
    if len(image_files) < 2:
        print(f"Not enough images for character {char_id}, found {len(image_files)}")
        return
    
    num_samples = min(num_samples, len(image_files))
    reference_file = image_files[0]
    reference_path = os.path.join(char_path, reference_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, num_samples):
        target_file = image_files[i]
        target_path = os.path.join(char_path, target_file)
        
        output_file = f"{char_id}_{i}.png"
        output_path = os.path.join(output_dir, output_file)
        
        print(f"Processing {target_file}...")
        test_identity_transfer(model, reference_path, target_path, output_path)
        metrics = compute_identity_metrics(model, reference_path, target_path)
        
        metrics_path = os.path.join(output_dir, f"{char_id}_{i}_metrics.txt")
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
    
    print(f"Processed {num_samples-1} images for character {char_id}")

def main():
    parser = argparse.ArgumentParser(description="Test identity preserving model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to model config')
    parser.add_argument('--reference', type=str, default=None, help='Path to reference image')
    parser.add_argument('--target', type=str, default=None, help='Path to target image')
    parser.add_argument('--output', type=str, default='output.png', help='Path to save output image')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset for batch testing')
    parser.add_argument('--bundle', type=str, default=None, help='Bundle ID for batch testing')
    parser.add_argument('--char', type=str, default=None, help='Character ID for batch testing')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples for batch testing')
    args = parser.parse_args()
    
    start_time = time.time()
    
    config = None
    if args.config:
        config = load_config(args.config)
    
    model = load_model(args.checkpoint, config)
    
    load_time = time.time()
    print(f"Model loading took {load_time - start_time:.2f} seconds")
    
    if args.reference and args.target:
        print(f"Testing identity transfer from {args.reference} to {args.target}")
        output_image = test_identity_transfer(model, args.reference, args.target, args.output)
        compute_identity_metrics(model, args.reference, args.target)
    elif args.dataset and args.bundle and args.char:
        print(f"Batch testing from dataset: {args.dataset}")
        output_dir = os.path.join(os.path.dirname(args.output), 'batch_output')
        batch_test_from_dataset(model, args.dataset, args.bundle, args.char, output_dir, args.num_samples)
    else:
        print("No test specified. Please provide either reference and target images, or dataset, bundle and char IDs.")
        print("Usage for single test: --reference <ref_img> --target <target_img>")
        print("Usage for batch test: --dataset <dataset_dir> --bundle <bundle_id> --char <char_id>")
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 