import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from train.architecture import IdentityPreservingFlux
except ImportError:
    try:
        from architecture import IdentityPreservingFlux
    except ImportError:
        sys.exit(1)

try:
    import wandb
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

class IdentityDataset(Dataset):
    
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
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_cache = use_cache
        self.min_images_per_char = min_images_per_char
        self.reference_image_idx = reference_image_idx
        
        if use_cache:
            self.cache_dir = Path(cache_dir or os.path.join(root_dir, "cache"))
            os.makedirs(self.cache_dir, exist_ok=True)
        
        if bundle_dirs is None:
            bundle_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        self.samples = []
        
        for bundle_dir in bundle_dirs:
            bundle_path = self.root_dir / bundle_dir
            
            char_dirs = [d for d in os.listdir(bundle_path) if os.path.isdir(bundle_path / d)]
            
            if max_chars_per_bundle and len(char_dirs) > max_chars_per_bundle:
                random.shuffle(char_dirs)
                char_dirs = char_dirs[:max_chars_per_bundle]
            
            for char_dir in tqdm(char_dirs, desc=f"Loading characters from bundle {bundle_dir}"):
                char_path = bundle_path / char_dir
                
                image_files = [f for f in os.listdir(char_path) if f.endswith('.jpg') or f.endswith('.png')]
                
                if len(image_files) < self.min_images_per_char:
                    continue
                
                pkl_path = os.path.join(char_path, 'char_data.pkl')
                if not os.path.exists(pkl_path):
                    continue
                
                image_files.sort()
                
                for img_file in image_files:
                    self.samples.append({
                        'char_path': str(char_path),
                        'img_file': img_file,
                        'char_id': f"{bundle_dir}_{char_dir}",
                        'is_reference': img_file == image_files[self.reference_image_idx],
                        'reference_file': image_files[self.reference_image_idx],
                        'pkl_path': pkl_path,
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        char_path = sample_info['char_path']
        img_file = sample_info['img_file']
        char_id = sample_info['char_id']
        reference_file = sample_info['reference_file']
        pkl_path = sample_info['pkl_path']
        
        target_img_path = os.path.join(char_path, img_file)
        target_img = Image.open(target_img_path).convert('RGB')
        
        ref_img_path = os.path.join(char_path, reference_file)
        ref_img = Image.open(ref_img_path).convert('RGB')
        
        if self.transform:
            target_img = self.transform(target_img)
            ref_img = self.transform(ref_img)
        
        try:
            with open(pkl_path, 'rb') as f:
                char_data = pickle.load(f)
            
            target_label = None
            for item in char_data:
                if os.path.basename(item.get('image_path', '')) == img_file:
                    target_label = item.get('prompt', f"A photo of character {char_id}")
                    break
                
            if target_label is None:
                target_label = f"A photo of character {char_id}"
        except Exception:
            target_label = f"A photo of character {char_id}"
        
        return {
            'reference_image': ref_img,
            'target_image': target_img,
            'source_image': ref_img,
            'char_id': char_id,
            'target_path': target_img_path,
            'reference_path': ref_img_path,
            'target_label': target_label,
        }

class IdentityContentLoss(nn.Module):
    def __init__(self, face_weight=1.0, body_weight=0.5, content_weight=0.5, device='cuda'):
        super().__init__()
        self.face_weight = face_weight
        self.body_weight = body_weight
        self.content_weight = content_weight
        self.device = device
        
        vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
        vgg16.eval()
        
        self.content_model = nn.Sequential(
            *list(vgg16.features.children())[:16]
        ).to(device)
        
        self.content_model.eval()
        for param in self.content_model.parameters():
            param.requires_grad = False
            
        self.content_model = self.content_model.float()
            
        self.mse_loss = nn.MSELoss()
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        
    def forward(self, ref_face_embed, ref_body_embed, output_face_embed, output_body_embed, output_image, target_image):
        loss_dict = {}
        
        face_loss = 0
        if ref_face_embed is not None and output_face_embed is not None:
            if not output_face_embed.requires_grad:
                ref_face_flat = ref_face_embed.view(ref_face_embed.size(0), -1).to(self.device).detach()
                output_face_flat = output_face_embed.view(output_face_embed.size(0), -1).to(self.device)
                if not output_face_flat.requires_grad:
                    output_face_flat = output_face_flat.detach().clone().requires_grad_(True)
            else:
                ref_face_flat = ref_face_embed.view(ref_face_embed.size(0), -1).to(self.device)
                output_face_flat = output_face_embed.view(output_face_embed.size(0), -1).to(self.device)
            
            face_sim = self.cosine_sim(ref_face_flat, output_face_flat)
            face_loss = torch.mean(1 - face_sim)
            loss_dict['face_loss'] = face_loss.item()
        
        body_loss = 0
        if ref_body_embed is not None and output_body_embed is not None:
            if not output_body_embed.requires_grad:
                ref_body_flat = ref_body_embed.view(ref_body_embed.size(0), -1).to(self.device).detach()
                output_body_flat = output_body_embed.view(output_body_embed.size(0), -1).to(self.device)
                if not output_body_flat.requires_grad:
                    output_body_flat = output_body_flat.detach().clone().requires_grad_(True)
            else:
                ref_body_flat = ref_body_embed.view(ref_body_embed.size(0), -1).to(self.device)
                output_body_flat = output_body_embed.view(output_body_embed.size(0), -1).to(self.device)
            
            body_sim = self.cosine_sim(ref_body_flat, output_body_flat)
            body_loss = torch.mean(1 - body_sim)
            loss_dict['body_loss'] = body_loss.item()
        
        content_loss = 0
        if self.content_weight > 0:
            if not output_image.requires_grad:
                output_image = output_image.detach().clone().requires_grad_(True)
            
            target_float = target_image.float().detach()
            output_float = output_image.float()
            
            with torch.no_grad():
                target_features = self.content_model(target_float)
            output_features = self.content_model(output_float)
            
            content_loss = self.mse_loss(output_features, target_features)
            loss_dict['content_loss'] = content_loss.item()
        
        total_loss = (
            self.face_weight * face_loss + 
            self.body_weight * body_loss + 
            self.content_weight * content_loss
        )
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp):
    model.set_training_mode(True)
    running_loss = 0.0
    all_loss_dicts = []
    
    for batch in tqdm(dataloader, desc="Training"):
        reference_images = batch['reference_image'].to(device)
        target_images = batch['target_image'].to(device)
        
        optimizer.zero_grad()
        
        model.extract_identity(reference_images)
        
        with autocast(enabled=use_amp):
            if hasattr(model, 'generate'):
                output_images = model.generate(
                    prompts=[""] * reference_images.size(0),
                    reference_images=reference_images,
                    guidance_scale=1.0,
                    num_inference_steps=1
                )
            else:
                output_images = model(target_images)
            
            ref_face_embedding = model.reference_face_embedding
            ref_body_embedding = model.reference_body_embedding
            
            output_face_embedding = model.output_face_embedding if hasattr(model, 'output_face_embedding') else None
            output_body_embedding = model.output_body_embedding if hasattr(model, 'output_body_embedding') else None
            
            loss, loss_dict = criterion(
                ref_face_embedding,
                ref_body_embedding,
                output_face_embedding,
                output_body_embedding,
                output_images,
                target_images
            )
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        all_loss_dicts.append(loss_dict)
    
    avg_loss = running_loss / len(dataloader)
    avg_loss_dict = {k: sum(d[k] for d in all_loss_dicts if k in d) / len(dataloader) for k in all_loss_dicts[0]}
    
    return avg_loss, avg_loss_dict

def validate(model, dataloader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    all_loss_dicts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            reference_images = batch['reference_image'].to(device)
            target_images = batch['target_image'].to(device)
            
            model.extract_identity(reference_images)
            
            if hasattr(model, 'generate'):
                output_images = model.generate(
                    prompts=[""] * reference_images.size(0),
                    reference_images=reference_images,
                    guidance_scale=1.0,
                    num_inference_steps=1
                )
            else:
                output_images = model(target_images)
            
            ref_face_embedding = model.reference_face_embedding
            ref_body_embedding = model.reference_body_embedding
            
            output_face_embedding = model.output_face_embedding if hasattr(model, 'output_face_embedding') else None
            output_body_embedding = model.output_body_embedding if hasattr(model, 'output_body_embedding') else None
            
            loss, loss_dict = criterion(
                ref_face_embedding,
                ref_body_embedding,
                output_face_embedding,
                output_body_embedding,
                output_images,
                target_images
            )
            
            running_loss += loss.item()
            all_loss_dicts.append(loss_dict)
    
    avg_loss = running_loss / len(dataloader)
    avg_loss_dict = {k: sum(d[k] for d in all_loss_dicts if k in d) / len(dataloader) for k in all_loss_dicts[0]}
    
    return avg_loss, avg_loss_dict

def visualize_examples(examples, epoch):
    plt.figure(figsize=(12, 10))
    for i, (ref_img, target_img, output_img) in enumerate(examples):
        plt.subplot(len(examples), 3, i * 3 + 1)
        plt.imshow(ref_img)
        plt.title(f"Reference {i+1}")
        plt.axis('off')
        
        plt.subplot(len(examples), 3, i * 3 + 2)
        plt.imshow(target_img)
        plt.title(f"Target {i+1}")
        plt.axis('off')
        
        plt.subplot(len(examples), 3, i * 3 + 3)
        plt.imshow(output_img)
        plt.title(f"Output {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

def identity_similarity_metrics(model, test_loader, device):
    model.eval()
    
    face_sims = []
    body_sims = []
    content_sims = []
    
    cosine_sim = nn.CosineSimilarity(dim=1)
    mse_loss = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing metrics"):
            reference_images = batch['reference_image'].to(device)
            target_images = batch['target_image'].to(device)
            
            model.extract_identity(reference_images)
            
            if hasattr(model, 'generate'):
                output_images = model.generate(
                    prompts=[""] * reference_images.size(0),
                    reference_images=reference_images,
                    guidance_scale=1.0,
                    num_inference_steps=1
                )
            else:
                output_images = model(target_images)
            
            ref_face_embed = model.reference_face_embedding
            ref_body_embed = model.reference_body_embedding
            
            output_face_embed = model.output_face_embedding if hasattr(model, 'output_face_embedding') else None
            output_body_embed = model.output_body_embedding if hasattr(model, 'output_body_embedding') else None
            
            if ref_face_embed is not None and output_face_embed is not None:
                ref_face_flat = ref_face_embed.view(ref_face_embed.size(0), -1)
                output_face_flat = output_face_embed.view(output_face_embed.size(0), -1)
                
                face_sim = cosine_sim(ref_face_flat, output_face_flat)
                face_sims.extend(face_sim.cpu().numpy())
            
            if ref_body_embed is not None and output_body_embed is not None:
                ref_body_flat = ref_body_embed.view(ref_body_embed.size(0), -1)
                output_body_flat = output_body_embed.view(output_body_embed.size(0), -1)
                
                body_sim = cosine_sim(ref_body_flat, output_body_flat)
                body_sims.extend(body_sim.cpu().numpy())
            
            pix_wise_mse = mse_loss(output_images, target_images).mean(dim=(1, 2, 3))
            content_sims.extend((-pix_wise_mse).cpu().numpy())
    
    metrics = {}
    if face_sims:
        metrics['face_similarity'] = np.mean(face_sims)
    if body_sims:
        metrics['body_similarity'] = np.mean(body_sims)
    if content_sims:
        metrics['content_similarity'] = np.mean(content_sims)
    
    return metrics

def init_model(input_size=512, device='cuda'):
    model = IdentityPreservingFlux(
        use_gpu=(device == 'cuda'),
        cache_dir=None
    )
    
    model = model.to(device)
    
    return model

def custom_collate_fn(batch):
    batch_dict = {
        'reference_image': torch.stack([item['reference_image'] for item in batch]),
        'target_image': torch.stack([item['target_image'] for item in batch]),
        'source_image': torch.stack([item['source_image'] for item in batch]),
        'char_id': [item['char_id'] for item in batch],
        'target_path': [item['target_path'] for item in batch],
        'reference_path': [item['reference_path'] for item in batch],
        'target_label': [item['target_label'] for item in batch],
    }
    
    return batch_dict

def visualize_results(model, dataloader, output_dir, device, epoch, max_samples=8, dpi=150):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    examples = []
    to_pil = transforms.ToPILImage()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_samples:
                break
                
            reference_images = batch['reference_image'].to(device)
            target_images = batch['target_image'].to(device)
            
            model.extract_identity(reference_images)
            
            if hasattr(model, 'generate'):
                output_images = model.generate(
                    prompts=[""] * reference_images.size(0),
                    reference_images=reference_images,
                    guidance_scale=1.0,
                    num_inference_steps=1
                )
            else:
                output_images = model(target_images)
            
            for j in range(min(len(reference_images), 1)):
                ref_img = to_pil(reference_images[j].cpu())
                target_img = to_pil(target_images[j].cpu())
                output_img = to_pil(output_images[j].cpu())
                
                examples.append((
                    np.array(ref_img),
                    np.array(target_img),
                    np.array(output_img)
                ))
                
                ref_img.save(os.path.join(output_dir, f"epoch_{epoch}_sample_{i}_reference.png"))
                target_img.save(os.path.join(output_dir, f"epoch_{epoch}_sample_{i}_target.png"))
                output_img.save(os.path.join(output_dir, f"epoch_{epoch}_sample_{i}_output.png"))
    
    comparison_fig = visualize_examples(examples, epoch)
    comparison_fig.savefig(os.path.join(output_dir, f"epoch_{epoch}_comparisons.png"), dpi=dpi)
    plt.close(comparison_fig)
    
    # Identity attention visualization
    if hasattr(model, 'visualize_attention'):
        try:
            attn_fig = model.visualize_attention()
            if attn_fig:
                attn_fig.savefig(os.path.join(output_dir, f"epoch_{epoch}_attention.png"), dpi=dpi)
                plt.close(attn_fig)
        except Exception:
            pass
    
    # If model has a method to save its own visualization
    if hasattr(model, 'save_visualization'):
        try:
            model.save_visualization(os.path.join(output_dir, f"epoch_{epoch}_model_viz.png"))
        except Exception:
            pass
    
    return examples

def plot_losses(train_losses, val_losses, output_dir, current_epoch, dpi=150):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    epochs = list(range(1, len(train_losses) + 1))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if len(train_losses) > 1:
        # Plot the recent trend in a separate graph
        window_size = min(10, len(train_losses))
        plt.subplot(1, 2, 2)
        plt.plot(epochs[-window_size:], train_losses[-window_size:], 'b-', label='Training Loss')
        if val_losses:
            plt.plot(epochs[-window_size:], val_losses[-window_size:], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Recent Loss Trend (Last {window_size} Epochs)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"loss_plot_epoch_{current_epoch}.png"), dpi=dpi)
    plt.close()

def main(config):
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Initialize WandB if enabled
    use_wandb = config.get('use_wandb', False)
    if use_wandb:
        try:
            run = wandb.init(
                project=config.get('wandb_project', 'identity-preserving-flux'),
                name=config.get('run_name', f"run_{int(time.time())}"),
                config=config
            )
        except Exception:
            use_wandb = False
    
    # Create transforms
    input_size = config.get('input_size', 512)
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    
    # Create dataset
    dataset = IdentityDataset(
        root_dir=config.get('data_dir', 'dataset'),
        bundle_dirs=config.get('bundle_dirs', None),
        max_chars_per_bundle=config.get('max_chars_per_bundle', 20),
        transform=transform,
        use_cache=config.get('use_cache', True),
        cache_dir=config.get('cache_dir', None),
        min_images_per_char=config.get('min_images_per_char', 3),
        reference_image_idx=config.get('reference_image_idx', 0)
    )
    
    # Split into train, val, test sets
    val_split = config.get('val_split', 0.1)
    test_split = config.get('test_split', 0.1)
    
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    test_size = int(dataset_size * test_split)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 2)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=device == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=device == 'cuda'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=device == 'cuda'
    )
    
    # Initialize model
    model = init_model(input_size=input_size, device=device)
    
    # Initialize criterion
    criterion = IdentityContentLoss(
        face_weight=config.get('face_weight', 1.0),
        body_weight=config.get('body_weight', 0.5),
        content_weight=config.get('content_weight', 0.5),
        device=device
    )
    
    # Initialize optimizer
    learning_rate = config.get('learning_rate', 0.0001)
    weight_decay = config.get('weight_decay', 0.00001)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=False
    )
    
    # Initialize amp scaler
    use_amp = config.get('use_amp', False) and device == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    
    num_epochs = config.get('num_epochs', 20)
    log_interval = config.get('log_interval', 5)
    save_interval = config.get('save_interval', 1)
    
    output_dir = config.get('output_dir', './checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        train_loss, train_loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        
        val_loss = None
        val_loss_dict = None
        
        if val_loader:
            val_loss, val_loss_dict = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        if val_loss is not None:
            val_losses.append(val_loss)
        
        epoch_time = time.time() - start_time
        
        if epoch % log_interval == 0 or epoch == num_epochs:
            plot_losses(train_losses, val_losses, output_dir, epoch)
        
        # Log metrics
        if use_wandb:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'epoch_time': epoch_time,
            }
            
            if train_loss_dict:
                for k, v in train_loss_dict.items():
                    log_dict[f'train_{k}'] = v
            
            if val_loss is not None:
                log_dict['val_loss'] = val_loss
                
                if val_loss_dict:
                    for k, v in val_loss_dict.items():
                        log_dict[f'val_{k}'] = v
            
            wandb.log(log_dict)
        
        # Save checkpoints
        if (epoch % save_interval == 0 or epoch == num_epochs):
            checkpoint_path = os.path.join(output_dir, f"epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, checkpoint_path)
            
            # Also save a safe tensors version
            safetensors_path = os.path.join(output_dir, f"epoch_{epoch}.safetensors")
            save_file(model.state_dict(), safetensors_path, metadata={"epoch": str(epoch)})
            
            # Visualize results
            _ = visualize_results(
                model, test_loader, os.path.join(output_dir, f"epoch_{epoch}"),
                device, epoch, max_samples=config.get('max_viz_samples', 8)
            )
        
        # Save best model
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, best_model_path)
            
            # Also save a safe tensors version
            best_safetensors_path = os.path.join(output_dir, "best_model.safetensors")
            save_file(model.state_dict(), best_safetensors_path, metadata={"epoch": str(epoch)})
    
    # Final evaluation on test set
    test_metrics = identity_similarity_metrics(model, test_loader, device)
    
    if use_wandb:
        wandb.log(test_metrics)
        wandb.finish()
    
    return model, test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train identity preserving model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config) 