import torch
from torch import nn
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from typing import NamedTuple
from collections.abc import Sequence, Hashable
from tqdm import tqdm
import numpy as np

# logger setup
from loguru import logger

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

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
# Device = torch.device("cpu")


class SequenceClassifier(nn.Module):

    class Input(NamedTuple):
        input_ids: torch.Tensor
        attention_masks: torch.Tensor

    class DataSplit(NamedTuple):
        train: DataLoader
        val: DataLoader
        test: DataLoader

    class UnpackedBatch(NamedTuple):
        input_ids: torch.Tensor
        input_masks: torch.Tensor
        labels: torch.Tensor

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **(model_kwargs or {})
        )
        self.model.to(Device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, **(tokenizer_kwargs or {})
        )

    def tokenize(
        self, samples: str | Sequence[str], max_length: int | None = None
    ) -> Input:
        if isinstance(samples, str):
            samples = [samples]

        input_ids = []
        attention_masks = []

        for sample in samples:
            encoded_dict = self.tokenizer(
                sample,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])

        return SequenceClassifier.Input(
            input_ids=torch.cat(input_ids, dim=0),
            attention_masks=torch.cat(attention_masks, dim=0),
        )

    @classmethod
    def convert_labels(cls, *labels: Sequence[Hashable]) -> torch.Tensor:
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

    @classmethod
    def train_validation_test_split(
        cls,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        labels: torch.Tensor | Sequence[torch.Tensor],
        batch_size: int = 32,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
    ) -> DataSplit:
        if train_size + val_size + test_size != 1:
            raise ValueError("train, validation and test size must sum up to 1.")
        if isinstance(labels, torch.Tensor):
            labels = [labels]
        dataset = TensorDataset(input_ids, attention_masks, *labels)
        train_set, val_set, test_set = random_split(
            dataset, [train_size, val_size, test_size]
        )

        return cls.DataSplit(
            DataLoader(
                train_set, sampler=RandomSampler(train_set), batch_size=batch_size
            ),
            DataLoader(val_set, batch_size=batch_size),
            DataLoader(test_set, batch_size=batch_size),
        )

    def perform_training(
        self,
        samples: Sequence[str],
        *labels: Sequence[Hashable],
        batch_size: int = 32,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        epochs: int = 4,
    ) -> None:
        if len({len(l) for l in labels}) != 1 or len(labels[0]) != len(samples):
            raise ValueError(
                "Each sequence of labels must have as many labels as there are samples."
            )

        # tokenize
        tokens = self.tokenize(samples)

        # convert labels to one hot
        one_hot = self.convert_labels(*labels)

        # get dataloaders
        data = self.train_validation_test_split(
            *tokens,
            one_hot,
            batch_size=batch_size,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )

        # get optimizer
        optimizer = AdamW(
            self.model.parameters(),
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
            self.train_epoch(data, optimizer, lr_scheduler)
            self.validate_epoch(data)

        logger.info("Training complete!")

    def train_epoch(self, data: DataSplit, optimizer, lr_scheduler) -> None:

        total_loss = 0

        # set model to train mode
        self.model.train()

        for batch in tqdm(
            data.train,
            position=1,
            desc="Batch training",
            unit="batch",
            leave=False,
        ):
            unpacked_batch = self._unpack_batch(batch)

            # clear any previously calculated gradients
            self.model.zero_grad()

            # forward pass
            output = self.model(
                unpacked_batch.input_ids,
                token_type_ids=None,
                attention_mask=unpacked_batch.input_masks,
                labels=unpacked_batch.labels,
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
            "Average training loss: {0:.2f}".format(total_loss / len(data.train))
        )

    def validate_epoch(self, data: DataSplit) -> None:

        total_loss = 0
        total_accuracy = 0

        # set model to eval mode
        self.model.eval()

        for batch in tqdm(
            data.val,
            desc="Batch validation",
            unit="batch",
            position=1,
            leave=False,
        ):

            unpacked_batch = self._unpack_batch(batch)

            # grads are only needed for backprop
            with torch.no_grad():

                output = self.model(
                    unpacked_batch.input_ids,
                    token_type_ids=None,
                    attention_mask=unpacked_batch.input_masks,
                    labels=unpacked_batch.labels,
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

        logger.info(f"Validation loss: {total_loss/len(data.val)}")
        logger.info(f"Validation accuracy: {total_accuracy / len(data.val)}")

    def _unpack_batch(self, batch: tuple[torch.Tensor, ...]) -> UnpackedBatch:
        return self.UnpackedBatch(
            batch[0].to(Device), batch[1].to(Device), batch[2].to(Device)
        )

    def _flat_accuracy(self, preds, labels) -> float:
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

