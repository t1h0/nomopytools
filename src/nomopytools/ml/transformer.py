import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.adamw import AdamW
from transformers import (
    AutoModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass as BaseAutoModel
from transformers.utils.generic import ModelOutput
from tqdm import tqdm
import numpy as np
from collections.abc import Sequence
from typing import Literal
from utils import train_validation_test_split, Device, to_device
from containers import DataSplit, ForwardSplitInput

# logger setup
from loguru import logger

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class Transformer(nn.Module):

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict | None = None,
        auto_model_class: type[BaseAutoModel] = AutoModel,
        *args,
        **kwargs,
    ) -> None:
        """Template for a Transformer model.

        Args:
            model_name (str): The model to use.
            model_kwargs (dict | None, optional): kwargs to pass to the model.
                Defaults to None.
            auto_model_class (type[BaseAutoModel], optional): AutoModel class
                to use for loading the model (e.g. AutoModelForSequenceClassification).
                Defaults to AutoModel.
            *args, **kwargs: To pass to nn.Module.
        """
        super().__init__(*args, **kwargs)
        self.model = auto_model_class.from_pretrained(
            model_name, **(model_kwargs or {})
        )
        self.to(Device)

    def forward(self, *args, **kwargs) -> ModelOutput:
        # to make things easier, we'll always return a ModelOutput
        kwargs["return_dict"] = True

        return self.model(*to_device(args), **to_device(kwargs))

    def infer(self, *args, **kwargs) -> ModelOutput:
        self.eval()

        with torch.no_grad():
            return self(*args, **kwargs)

    def perform_training(
        self,
        forward_inputs: ForwardSplitInput | Sequence[ForwardSplitInput],
        forward_kwargs: dict | None = None,
        batch_size: int = 32,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        epochs: int = 4,
        *args,
        **kwargs,
    ) -> None:
        """Trains the classifier.

        Args:
            forward_inputs (ForwardSplitInput | Sequence[ForwardSplitInput]):
                ForwardSplitInput tuple(s) that will be train-validation-test split and
                passed to forward call. E.g. input_ids and attention_mask.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 32.
            train_size (float, optional): Train set size (as proportion).
                Defaults to 0.8.
            val_size (float, optional): Validation set size (as proportion).
                Defaults to 0.1.
            test_size (float, optional): Test set size (as proportion).
                Defaults to 0.1.
            epochs (int, optional): Number of epochs to run. Defaults to 4.
        """
        if isinstance(forward_inputs, ForwardSplitInput):
            forward_inputs = (forward_inputs,)
        # get forward data keys
        forward_data_keys = tuple(inp.key for inp in forward_inputs)

        # get dataloaders
        data = train_validation_test_split(
            *(inp.tensor for inp in forward_inputs),
            batch_size=batch_size,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )

        # get optimizer
        optimizer = AdamW(
            self.parameters(),
            # lr=2e-5,
        )

        # get learning rate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # Default value in run_glue.py
            num_training_steps=len(data.train) * epochs,
        )

        # train

        for epoch in tqdm(
            range(epochs), desc="Epochs", unit="epoch", position=0, leave=True
        ):
            logger.info(f"=== Epoch {epoch} ===")
            self.train_epoch(
                data,
                forward_data_keys,
                optimizer,
                lr_scheduler,
                forward_kwargs,
            )
            self.eval_epoch("validation", data, forward_data_keys, forward_kwargs)
            self.eval_epoch("test", data, forward_data_keys, forward_kwargs)

        logger.info("Training complete!")

    def train_epoch(
        self,
        data: DataSplit,
        data_keys: Sequence[str],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        forward_kwargs: dict | None = None,
    ) -> None:
        """Trains one epoch.

        Args:
            data (DataSplit): Train, validation and test sets.
            data_keys (Sequence[str]): Keys of each data Tensor to pass it
                to forward call with.
            optimizer (Optimizer): The optimizer to use.
            lr_scheduler (LRScheduler): The learning rate scheduler to use.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.
        """

        total_loss = 0

        # set model to train mode
        self.train()

        for batch in tqdm(
            data.train,
            position=1,
            desc="Batch training",
            unit="batch",
            leave=False,
        ):

            # clear any previously calculated gradients
            self.zero_grad()

            # forward pass
            output = self(
                **dict(zip(data_keys, batch)),
                **(forward_kwargs or {}),
            )

            # get loss
            loss = output.loss
            total_loss += loss.item()

            # backward pass
            loss.backward()

            # Clip norm of the gradients to 1.0 to prevent
            # exploding gradients problem
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update parameters
            optimizer.step()

            # update learning rate
            lr_scheduler.step()

        logger.info(
            "average training loss: {0:.2f}".format(total_loss / len(data.train))
        )

    def eval_epoch(
        self,
        eval_type: Literal["validation", "test"],
        data: DataSplit,
        data_keys: Sequence[str],
        forward_kwargs: dict | None = None,
    ) -> None:
        """Evaluates one epoch.

        Args:
            eval_type ("validation" | "test"): The type of evaluation and subsequently
                the dataset to use.
            data (DataSplit): Train, validation and test sets.
            data_keys (Sequence[str]): Keys of each data Tensor to pass it
                to forward call with.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.

        """

        total_loss = 0
        total_accuracy = 0

        self.eval()

        for batch in tqdm(
            data.test if eval_type == "test" else data.val,
            desc=f"Batch {eval_type}",
            unit="batch",
            position=1,
            leave=False,
        ):

            with torch.no_grad():
                output = self(
                    **dict(zip(data_keys, batch)),
                    **(forward_kwargs or {}),
                )

            loss = output.loss
            logits = output.logits

            # Accumulate the validation loss.
            total_loss += loss.item()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_accuracy += self._flat_accuracy(
                logits.detach().cpu().numpy(), batch[2].numpy()
            )

        logger.info("{0} loss: {1:.2f}".format(eval_type, total_loss / len(data.val)))
        logger.info(
            "{0} accuracy: {1:.2f}".format(eval_type, total_accuracy / len(data.val))
        )

    def _flat_accuracy(self, preds, labels) -> float:
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)