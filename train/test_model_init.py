import os
import sys
import torch
import logging

# Add parent directory to path to import from train directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    # Try importing the model
    from train.architecture import IdentityPreservingFlux
    logger.info("Successfully imported IdentityPreservingFlux")
except ImportError as e:
    try:
        # Try relative import
        from architecture import IdentityPreservingFlux
        logger.info("Successfully imported IdentityPreservingFlux (relative import)")
    except ImportError as e:
        logger.error(f"Error importing IdentityPreservingFlux: {e}")
        sys.exit(1)

def test_model_init():
    """Test model initialization and training mode changes"""
    logger.info("Initializing model")
    
    try:
        # Initialize model
        model = IdentityPreservingFlux()
        logger.info("Model successfully initialized")
        
        # Test training mode methods
        logger.info(f"Initial training_mode: {model.training_mode}")
        
        # Test set_training_mode
        model.set_training_mode(False)
        logger.info(f"After set_training_mode(False): {model.training_mode}")
        
        # Test train
        model.train()
        logger.info(f"After train(): {model.training_mode}")
        
        # Test eval
        model.eval()
        logger.info(f"After eval(): {model.training_mode}")
        
        logger.info("All training mode tests passed")
        return True
    except Exception as e:
        logger.error(f"Error during model initialization test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Running model initialization test")
    success = test_model_init()
    
    if success:
        logger.info("Model initialization test completed successfully")
        sys.exit(0)
    else:
        logger.error("Model initialization test failed")
        sys.exit(1) 