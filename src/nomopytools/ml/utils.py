import torch
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
from loguru import logger
from collections.abc import Sequence, Hashable, Iterable
from itertools import islice
from bidict import bidict
from typing import Any, TypeVar, overload
from datetime import datetime
from .containers import DataSplit

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

TensorContainer = TypeVar(
    "TensorContainer", dict[Any, Any], Sequence[Any], torch.Tensor
)


def to_device(
    cont: TensorContainer, device: torch.device | None = None
) -> TensorContainer:
    """Transfers a Tensor or all tensor inside a list or inside a dictionary to Device.

    Args:
        cont (TensorContainer): A dictionary, list or torch.Tensor.
        device (torch.device | None, optional): Torch device to use. If None, will
            select GPU if possible, else CPU. Defaults to None.

    Returns:
        TensorContainer: The input with all containing Tensors transferred to Device.
    """
    device = device or Device

    if isinstance(cont, Sequence):
        return [to_device(s, device) for s in cont]

    elif isinstance(cont, torch.Tensor):
        return cont.to(device)

    elif isinstance(cont, dict):
        return {k: to_device(v, device) for k, v in cont.items()}

    else:
        return cont


def train_validation_test_split(
    *tensors: torch.Tensor,
    batch_size: int = 32,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_seed: int | None = None,
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
        random_seed (int, optional). Seed to use for randomization. If None, will let
            torch decide. Defaults to None.

    Raises:
        ValueError: If train, validation and test size don't sum up to 1.

    Returns:
        SequenceClassifier.DataSplit: The three sets.
    """
    if train_size + val_size + test_size != 1:
        raise ValueError("train, validation and test size must sum up to 1.")
    dataset = TensorDataset(*tensors)
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=(
            torch.Generator().manual_seed(random_seed)
            if random_seed
            else torch.Generator()
        ),
    )

    return DataSplit(
        DataLoader(
            train_set,
            sampler=RandomSampler(
                train_set,
                generator=(
                    torch.Generator().manual_seed(random_seed)
                    if random_seed
                    else torch.Generator()
                ),
            ),
            batch_size=batch_size,
        ),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size),
    )


@overload
def convert_labels(
    *labels: Sequence[Hashable],
) -> tuple[tuple[bidict[int, Hashable], ...], torch.Tensor]: ...


@overload
def convert_labels() -> None: ...


def convert_labels(
    *labels: Sequence[Hashable],
) -> tuple[tuple[bidict[int, Hashable], ...], torch.Tensor] | None:
    """Convert labels into a vector ready for loss computation.

    Args:
        *labels (Sequence[Hashable]): One sequence of labels per task.

    Returns:
        tuple[bidict[int, Hashable], ...] , torch.Tensor: Tuple of one bidict
            for index <> label per input label sequence and the converted labels.
    """
    if not labels:
        return None

    out_bidicts = []
    out_tensors = []

    for label_set in labels:
        # mapping: label -> index
        label_to_index = bidict(
            {index: label for index, label in enumerate({*label_set})}
        )

        # convert: label -> index
        indices = [label_to_index.inverse[label] for label in label_set]

        out_bidicts.append(label_to_index)
        out_tensors.append(indices)

    if len(out_tensors) == 1:
        out_tensors = torch.Tensor(out_tensors[0])
    else:
        out_tensors = torch.Tensor(out_tensors).transpose(0, 1)

    return tuple(out_bidicts), out_tensors.to(torch.int64)


def one_hot_multitask(
    tensor: torch.IntTensor, num_classes: Sequence[int] | None = None
) -> torch.Tensor:
    """Create one hot mappings for multi task true labels. E.g. a sample with true label
    class 1 for the first task and 3 for the second will get [0,1,0,0,0,1].

    Args:
        tensor (torch.IntTensor): Tensor with samples as rows and true labels per task
            as columns.
        num_classes (Sequence[int] | None, optional): Number of classes per task.
            If None, will take max values found in tensor. Defaults to None.

    Returns:
        torch.Tensor: The one hot tensor with samples as rows their one hot vectors
            per class strung together.
    """
    if num_classes is None:
        # add one because of zero indexing
        num_classes = (tensor.max(0)[0] + 1).tolist()

    return torch.cat(
        tuple(one_hot(t, n) for t, n in zip(tensor.transpose(0, 1), num_classes)), 1
    )


IterYield = TypeVar("IterYield")


def batched(iterable: Iterable[IterYield], n: int):
    """Adaptation of itertools.batched (python >=3.12)

    Args:
        iterable (Iterable[IterYield]): Iterable to batch.
        n (int): Batch size.

    Raises:
        ValueError: If batch size < 1.

    Yields:
        list[IterYield,...]: The batches.
    """
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch


def get_datetime() -> str:
    """Get current datetime.

    Returns:
        str: The datetime as a string in the format YYYY-MM-DD_HH-MM-SS-MSSS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
