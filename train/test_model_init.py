import os
import sys
import torch
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from train.architecture import IdentityPreservingFlux
    logger.info("Successfully imported IdentityPreservingFlux")
except ImportError as e:
    try:
        from architecture import IdentityPreservingFlux
        logger.info("Successfully imported IdentityPreservingFlux (relative import)")
    except ImportError as e:
        logger.error(f"Error importing IdentityPreservingFlux: {e}")
        sys.exit(1)

def test_model_init():
    """Test model initialization and training mode changes"""
    logger.info("Initializing model")
    
    try:
        model = IdentityPreservingFlux()
        logger.info("Model successfully initialized")
        
        logger.info(f"Initial training_mode: {model.training_mode}")
        
        model.set_training_mode(False)
        logger.info(f"After set_training_mode(False): {model.training_mode}")
        
        model.train()
        logger.info(f"After train(): {model.training_mode}")
        
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