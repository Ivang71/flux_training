# DTBack: Identity-Preserving Flux Training

A training pipeline for identity preservation in the Flux image generation model. This repository contains code to train and use identity preservation models that maintain a person's facial and body features in images generated with specific prompts.

## Features

- Production-ready training pipeline for identity preservation
- Dataset handling for identity extraction
- Robust model architecture with face and body identity injection
- Visualization tools to track training progress
- Configurable training parameters

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU
- PyTorch 2.0+

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/dtback.git
cd dtback
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Download pre-trained models:
```bash
python scripts/download_models.py
```

## Dataset Preparation

The training pipeline expects a dataset organized by character, with multiple images per character:

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

## Training

Training the identity preservation model:

```bash
python train/run_training.py --config train/config.yaml
```

### Configuration

The `train/config.yaml` file contains all training parameters:

```yaml
# Dataset parameters
data_dir: "dataset"
max_chars_per_bundle: 20
min_images_per_char: 3

# Model parameters
use_simple_model: true
input_size: 512
num_face_latents: 16
num_body_latents: 16

# Training parameters
batch_size: 2
num_epochs: 30
learning_rate: 0.0001
weight_decay: 0.00001
face_weight: 1.0
body_weight: 0.5
content_weight: 0.5

# Output parameters
output_dir: "checkpoints/identity_preserving"
```

## Inference

To use the trained model for inference:

```bash
python inference/generate.py --model checkpoints/identity_preserving/best_model.pt \
                             --reference path_to_reference_image.jpg \
                             --prompt "A photo of a person in Paris" \
                             --output output_directory
```

## Model Architecture

The Identity-Preserving Flux model consists of:

1. Face Identity Extractor: Extracts facial features using InsightFace
2. Body Identity Extractor: Extracts body features from the input image
3. Identity Injection Mechanism: Injects identity features into the transformer blocks of the Flux model
4. Training Loop: Measures and minimizes identity differences between reference and generated images

## Results

The training pipeline produces:

1. Trained model checkpoints saved in the `output_dir` 
2. Visualizations showing identity preservation progress
3. Logs with training and validation metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Flux model from Black Forest Labs
- InsightFace for face recognition capabilities
- PyTorch for the deep learning framework 