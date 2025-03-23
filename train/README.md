# Identity-Preserving Flux Training Pipeline

This directory contains the training pipeline for the Identity-Preserving Flux model, which learns how to preserve a person's identity across images.

## Overview

The Identity-Preserving model is designed to maintain a person's identity features when generating images with specific prompts. The training pipeline includes:

1. A dataset loader that pairs reference images with target images
2. Identity extraction capabilities using InsightFace
3. A trainable autoencoder model for identity preservation
4. Loss functions that measure identity similarity and content preservation

## File Structure

- `run_training.py`: Main entry point for training
- `train_identity_preserving.py`: Core training loop and model definitions
- `architecture.py`: Implementation of the IdentityPreservingFlux model
- `config.yaml`: Configuration file with training parameters

## Dataset Format

The training dataset should be structured as follows:

```
dataset/
  ├── 0/
  │   ├── char_00001/
  │   │   ├── img_00001.jpg
  │   │   ├── img_00002.jpg
  │   │   └── ...
  │   └── ...
  ├── 1/
  │   └── ...
  └── ...
```

Each character subdirectory should contain multiple images of the same person, with the first image (by alphabetical order) used as the reference image.

## Training Process

The training process follows these steps:

1. Load dataset and split into train/validation/test sets
2. Initialize the SimpleIdentityAutoencoder or IdentityPreservingFlux model
3. For each epoch:
   - Extract identity embeddings from reference images
   - Generate images using the model
   - Compare identity between reference and generated images
   - Update model weights to minimize identity difference

## Configuration

The training can be configured using the `config.yaml` file. Important parameters include:

- `data_dir`: Path to the dataset directory
- `use_simple_model`: Whether to use the SimpleIdentityAutoencoder (for training) or full Flux model
- `batch_size`: Number of samples per batch
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization
- `face_weight`: Weight for face identity loss component
- `content_weight`: Weight for content preservation loss

## Usage

To start training:

```bash
python3 train/run_training.py --config train/config.yaml
```

The training script will:
1. Load the dataset
2. Initialize the model
3. Train for the specified number of epochs
4. Save checkpoint models and visualizations

## Results

Training produces the following outputs:

1. Model checkpoints saved in the specified `output_dir`
2. Visualizations showing identity preservation progress at each epoch
3. Logs with training and validation metrics

## Extending

To extend this pipeline:
- Add new identity feature extractors in `architecture.py`
- Implement custom loss functions in `train_identity_preserving.py`
- Modify the dataset loader for different data formats

## Requirements

- PyTorch
- torchvision
- matplotlib
- InsightFace (for face analysis)
- CUDA-capable GPU (recommended) 