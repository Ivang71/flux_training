import os
import sys
import yaml
import argparse
import logging
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run_training.log")
    ]
)
logger = logging.getLogger(__name__)
def convert_to_correct_type(key, value):
    if isinstance(value, str):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        if key in ['lr', 'learning_rate', 'weight_decay', 'dropout', 
                   'face_weight', 'body_weight', 'content_weight']:
            try:
                return float(value)
            except ValueError:
                pass
        if key in ['batch_size', 'num_epochs', 'num_workers', 'save_interval', 
                  'log_interval', 'val_interval', 'max_images', 'seed']:
            try:
                return int(value)
            except ValueError:
                pass
    return value
def main():
    parser = argparse.ArgumentParser(description="Run identity preserving training")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--override', nargs='*', help='Override config values (format: key1=value1 key2=value2)')
    args = parser.parse_args()
    try:
        logger.info(f"Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded {len(config)} configuration parameters")
        if args.override:
            logger.info(f"Applying {len(args.override)} overrides")
            for override in args.override:
                try:
                    key, value = override.split('=')
                    typed_value = convert_to_correct_type(key, value)
                    logger.info(f"Overriding {key}: {config.get(key, 'NOT_FOUND')} -> {typed_value} (type: {type(typed_value).__name__})")
                    config[key] = typed_value
                except ValueError:
                    logger.error(f"Invalid override format: {override} (should be key=value)")
        if 'weight_decay' in config:
            logger.info(f"Weight decay: {config['weight_decay']} (type: {type(config['weight_decay']).__name__})")
        if 'lr' in config:
            logger.info(f"Learning rate: {config['lr']} (type: {type(config['lr']).__name__})")
        for key in ['batch_size', 'num_epochs', 'num_workers']:
            if key in config and not isinstance(config[key], int):
                config[key] = int(config[key])
                logger.info(f"Converted {key} to int: {config[key]}")
        for key in ['lr', 'weight_decay', 'face_weight', 'body_weight', 'content_weight']:
            if key in config and not isinstance(config[key], float):
                config[key] = float(config[key])
                logger.info(f"Converted {key} to float: {config[key]}")
        for key in ['use_wandb', 'use_amp']:
            if key in config and not isinstance(config[key], bool):
                if isinstance(config[key], str):
                    config[key] = config[key].lower() == 'true'
                logger.info(f"Converted {key} to bool: {config[key]}")
        try:
            try:
                from train.train_identity_preserving import main as train_main
                logger.info("Imported train_main function from train.train_identity_preserving")
            except ImportError:
                from train_identity_preserving import main as train_main
                logger.info("Imported train_main function from train_identity_preserving (relative import)")
        except ImportError as e:
            logger.error(f"Could not import main function from train_identity_preserving.py: {e}")
            logger.error(traceback.format_exc())
            return
        try:
            try:
                from train.architecture import IdentityPreservingFlux
                logger.info("Successfully imported IdentityPreservingFlux from train.architecture")
            except ImportError:
                try:
                    from architecture import IdentityPreservingFlux
                    logger.info("Successfully imported IdentityPreservingFlux from architecture (relative import)")
                except ImportError:
                    logger.warning("Could not import IdentityPreservingFlux directly")
                    IdentityPreservingFlux = None
            if IdentityPreservingFlux:
                dummy_model = IdentityPreservingFlux()
                if hasattr(dummy_model, 'set_training_mode'):
                    logger.info("Model has custom set_training_mode method - will use this instead of train()")
                else:
                    logger.warning("Model does not have set_training_mode method - may have conflicts with YOLO")
        except Exception as e:
            logger.warning(f"Error testing model architecture: {e}")
            logger.warning("Will proceed with training, but there might be issues with train() method")
        output_dir = config.get('output_dir')
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        logger.info("Starting training...")
        try:
            train_main(config)
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            logger.error(traceback.format_exc())
            error_str = str(e)
            if "'bool' object is not callable" in error_str:
                logger.error("This error is likely due to a collision between your model's train() method and YOLOv8's train() method.")
                logger.error("Fix: Use the set_training_mode() method instead of train() in your model and training loop.")
            elif "CUDA out of memory" in error_str:
                logger.error("CUDA out of memory error. Try reducing batch size or model size.")
    except Exception as e:
        logger.error(f"Error running training: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
if __name__ == "__main__":
    main() 