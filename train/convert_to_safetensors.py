import argparse
import os
import torch
from safetensors.torch import save_file
import glob
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('convert_to_safetensors')

def convert_checkpoint(input_path, output_path):
    """
    Convert a PyTorch checkpoint (.pt or .pth) to safetensors format.
    
    Args:
        input_path: Path to the input PyTorch checkpoint
        output_path: Path to save the safetensors checkpoint
    """
    logger.info(f"Converting {input_path} to {output_path}")
    
    # Load the PyTorch checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Prepare tensors and metadata
    metadata = {}
    tensors = {}
    
    for k, v in checkpoint.items():
        if isinstance(v, torch.Tensor):
            tensors[k] = v
        elif k == 'model_state_dict' or k == 'optimizer_state_dict':
            for param_key, param_value in v.items():
                if isinstance(param_value, torch.Tensor):
                    tensors[f"{k}.{param_key}"] = param_value
        else:
            # Convert scalar values to strings for metadata
            metadata[k] = str(v)
    
    # Save as safetensors file
    save_file(tensors, output_path, metadata=metadata)
    logger.info(f"Successfully converted to {output_path}")
    
    # Verify file size
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    converted_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    logger.info(f"Original size: {original_size:.2f} MB, Converted size: {converted_size:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch checkpoints to safetensors format')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input PyTorch checkpoint file or directory containing .pt files')
    parser.add_argument('--output', type=str,
                        help='Output safetensors file or directory (if input is a directory)')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively search for .pt files in subdirectories')
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Single file conversion
        if not args.output:
            args.output = os.path.splitext(args.input)[0] + '.safetensors'
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        convert_checkpoint(args.input, args.output)
    elif os.path.isdir(args.input):
        # Directory conversion
        if not args.output:
            args.output = args.input
        os.makedirs(args.output, exist_ok=True)
        
        # Find all .pt files
        if args.recursive:
            pt_files = glob.glob(os.path.join(args.input, '**', '*.pt'), recursive=True)
            pt_files += glob.glob(os.path.join(args.input, '**', '*.pth'), recursive=True)
        else:
            pt_files = glob.glob(os.path.join(args.input, '*.pt'))
            pt_files += glob.glob(os.path.join(args.input, '*.pth'))
        
        if not pt_files:
            logger.warning(f"No .pt or .pth files found in {args.input}")
            return
        
        logger.info(f"Found {len(pt_files)} PyTorch checkpoint files to convert")
        
        for pt_file in tqdm(pt_files):
            # Determine output path
            rel_path = os.path.relpath(pt_file, args.input)
            output_path = os.path.join(args.output, os.path.splitext(rel_path)[0] + '.safetensors')
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert the file
            convert_checkpoint(pt_file, output_path)
    else:
        logger.error(f"Input path {args.input} does not exist")
        return

if __name__ == "__main__":
    main() 