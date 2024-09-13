import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
from loguru import logger
from collections.abc import Sequence, Hashable
from containers import DataSplit
from typing import Any, TypeVar

# Device inspection
if torch.cuda.is_available():
    Device = torch.device("cuda")
    logger.info("There are %d GPU(s) available." % torch.cuda.device_count())
    logger.info("Will use GPU:", torch.cuda.get_device_name(0))

elif torch.backends.mps.is_available():
    Device = torch.device("mps")
    logger.info("Will use MPS.")

else:
    logger.info("No GPU available, using the CPU instead.")
    Device = torch.device("cpu")

TensorContainer = TypeVar("TensorContainer", dict[Any, Any], Sequence[Any], torch.Tensor)


def to_device(cont: TensorContainer) -> TensorContainer:
    """Transfers a Tensor or all tensor inside a list or inside a dictionary to Device.

    Args:
        cont (TensorContainer): A dictionary, list or torch.Tensor.

    Returns:
        TensorContainer: The input with all containing Tensors transferred to Device.
    """

    if isinstance(cont, Sequence):
        return [to_device(s) for s in cont]

    elif isinstance(cont, torch.Tensor):
        return cont.to(Device)

    elif isinstance(cont, dict):
        return {k: to_device(v) for k, v in cont.items()}

    else:
        return cont


def train_validation_test_split(
    *tensors: torch.Tensor,
    batch_size: int = 32,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> DataSplit:
    """Create a train / validation / test split.

    Args:
        *tensors (torch.Tensor): The tensors to each split and return in the DataLoaders.
        batch_size (int, optional): Batch size. Defaults to 32.
        train_size (float, optional): Train set size (as proportion).
            Defaults to 0.8.
        val_size (float, optional): Validation set size (as proportion).
            Defaults to 0.1.
        test_size (float, optional): Test set size (as proportion).
            Defaults to 0.1.

    Raises:
        ValueError: If train, validation and test size don't sum up to 1.

    Returns:
        SequenceClassifier.DataSplit: The three sets.
    """
    if train_size + val_size + test_size != 1:
        raise ValueError("train, validation and test size must sum up to 1.")
    dataset = TensorDataset(*tensors)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )

    return DataSplit(
        DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size),
    )


def convert_labels(*labels: Sequence[Hashable]) -> torch.Tensor:
    """Convert labels into a vector ready for loss computation.

    Returns:
        torch.Tensor: The converted labels.
    """
    out = []

    for label_set in labels:
        # mapping: label -> index
        label_to_index = {label: index for index, label in enumerate({*label_set})}

        # convert: label -> index
        indices = [label_to_index[label] for label in label_set]

        out.append(indices)

    if len(out) == 1:
        out = torch.Tensor(out[0])
    else:
        out = torch.Tensor(out).transpose(0, 1)

    return out.to(torch.int64)
