from typing import NamedTuple
from torch.utils.data import DataLoader


class DataSplit(NamedTuple):
    """DataLoaders for train/validation/test split"""

    train: DataLoader
    val: DataLoader
    test: DataLoader
