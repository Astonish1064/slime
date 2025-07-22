
_ROOT_LOGGER_PATH = "/root/data/experiments/log"

def torch_available():
    """
    Check if PyTorch is available in the environment.
    
    Returns:
        bool: True if PyTorch is available, False otherwise.
    """
    try:
        import torch
        import torch.distributed

        return True
    except ImportError:
        return False


