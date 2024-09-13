from typing import NamedTuple
import torch
from torch.utils.data import DataLoader


class DataSplit(NamedTuple):
    """DataLoaders for train/validation/test split"""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    
class ForwardSplitInput(NamedTuple):
    """Transformer input Tensor that should be train-validation-test split.
    
    Args:
        key (str): Key of the Tensor to pass it to forward call with.
        tensor (torch.Tensor): The actual Tensor.
    """
    
    key: str
    tensor: torch.Tensor    