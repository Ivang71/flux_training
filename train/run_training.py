import os
import sys
import yaml
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        if args.override:
            for override in args.override:
                try:
                    key, value = override.split('=')
                    typed_value = convert_to_correct_type(key, value)
                    config[key] = typed_value
                except ValueError:
                    pass
        
        for key in ['batch_size', 'num_epochs', 'num_workers']:
            if key in config and not isinstance(config[key], int):
                config[key] = int(config[key])
        
        for key in ['lr', 'weight_decay', 'face_weight', 'body_weight', 'content_weight']:
            if key in config and not isinstance(config[key], float):
                config[key] = float(config[key])
        
        for key in ['use_wandb', 'use_amp']:
            if key in config and not isinstance(config[key], bool):
                if isinstance(config[key], str):
                    config[key] = config[key].lower() == 'true'
        
        try:
            try:
                from train.train_identity_preserving import main as train_main
            except ImportError:
                from train_identity_preserving import main as train_main
        except ImportError:
            return
        
        try:
            try:
                from train.architecture import IdentityPreservingFlux
            except ImportError:
                try:
                    from architecture import IdentityPreservingFlux
                except ImportError:
                    IdentityPreservingFlux = None
            
            if IdentityPreservingFlux:
                dummy_model = IdentityPreservingFlux()
        except Exception:
            pass
        
        output_dir = config.get('output_dir')
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            train_main(config)
        except Exception as e:
            error_str = str(e)
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main() 