import torch
from torch.utils.data import DataLoader
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import NamedTuple, Generic, TypeVar


class DataSplit(NamedTuple):
    """DataLoaders for train/validation/test split"""

    train: DataLoader
    validate: DataLoader
    test: DataLoader


OutputHeadKey = TypeVar("OutputHeadKey", bound=str)


@dataclass
class MultiLabelOutput(ModelOutput, Generic[OutputHeadKey]):
    """Dataclass for multi-label output."""

    loss: torch.FloatTensor | None = None
    losses: dict[OutputHeadKey, torch.FloatTensor] | None = None
    logits: dict[OutputHeadKey, torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Metric:
    """Dataclass for output metrics."""

    name: str
    value: float
