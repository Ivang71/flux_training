# Identity Preservation Training Pipeline

This directory contains scripts for training and testing an identity preservation model for the Flux image generation system.

## Overview

The identity preservation model learns to preserve facial and body identity features when generating images. It works by extracting identity embeddings from reference images and injecting them into the Flux model.

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.13+
- CUDA-compatible GPU (recommended)
- Diffusers library with Flux support

### Installation

Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install diffusers transformers insightface ultralytics
pip install tqdm matplotlib pillow wandb pyyaml
```

## Dataset Structure

The dataset should follow this structure:

```
dataset_creation/data/dataset/
├── 0/
│   ├── 20/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── char_data.pkl (can be ignored)
│   ├── 21/
│   │   └── ...
│   └── ...
├── 1/
│   └── ...
└── ...
```

Each character directory contains multiple images of the same individual.

## Training

### Configuration

The training parameters are specified in the `config.yaml` file. You can modify this file to change the training settings.

### Running Training

To train the model with the default configuration:

```bash
python train/run_training.py --config train/config.yaml
```

To override specific configuration parameters:

```bash
python train/run_training.py --config train/config.yaml --override batch_size=8 num_epochs=100
```

Alternatively, you can run the training script directly:

```bash
python train/train_identity_preserving.py --config train/config.yaml
```

### Training Parameters

Key parameters in the config file:

- **Dataset parameters**: `data_dir`, `bundle_dirs`, `max_chars_per_bundle`, `min_images_per_char`
- **Model parameters**: `num_face_latents`, `num_body_latents`, `num_fused_latents`, `use_identity_fusion`
- **Training parameters**: `batch_size`, `num_epochs`, `lr`, `face_weight`, `body_weight`, `content_weight`
- **Monitoring**: `use_wandb`, `wandb_project`

## Testing

After training, you can test the model using the test script:

```bash
python train/test_model.py --checkpoint ./checkpoints/identity_preserving/identity_preserving_v1_best.pt --config train/config.yaml --reference path/to/reference.jpg --target path/to/target.jpg
```

For batch testing on the dataset:

```bash
python train/test_model.py --checkpoint ./checkpoints/identity_preserving/identity_preserving_v1_best.pt --dataset dataset_creation/data/dataset --bundle 0 --char 20 --output_dir ./test_outputs
```

## Monitoring

If `use_wandb` is enabled in the config, training progress will be logged to [Weights & Biases](https://wandb.ai/). This includes:

- Loss curves
- Example outputs
- Identity metrics
- Hyperparameters

## Hyperparameters

Recommended hyperparameters:

- **Learning rate**: 1e-4 to 5e-5
- **Batch size**: 4-8 (depending on GPU memory)
- **Epochs**: 50-100
- **Identity weights**: face_weight=1.0, body_weight=0.5
- **Content weight**: 0.5

## Troubleshooting

- **Out of memory errors**: Reduce batch size or image resolution
- **No faces detected**: Ensure reference images have clear, frontal faces
- **Poor identity preservation**: Increase face_weight or adjust num_face_latents
- **Training takes too long**: Reduce the number of bundles or characters per bundle

## Model Outputs

The training process saves:

- Best model checkpoint (`identity_preserving_v1_best.pt`)
- Latest model checkpoint (`identity_preserving_v1_latest.pt`)
- Training logs
- Example outputs (if using wandb) 