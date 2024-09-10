import torch
from loguru import logger

if torch.cuda.is_available():    
    Device = torch.device("cuda")
    logger.debug('There are %d GPU(s) available.' % torch.cuda.device_count())
    logger.debug('Will use GPU:', torch.cuda.get_device_name(0))

elif torch.backends.mps.is_available():
    Device = torch.device("mps")
    logger.debug("Will use MPS.")

else:
    logger.debug('No GPU available, using the CPU instead.')
    Device = torch.device("cpu")
