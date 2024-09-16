import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForTextEncoding,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutput
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm
from bidict import bidict
from dataclasses import dataclass
from collections.abc import Sequence, Hashable
import math
from typing import TypeVar, Generic
from .utils import batched, convert_labels, one_hot_multitask
from .transformer import Transformer, BaseAutoModel, logger


class TextTransformer(Transformer):

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        auto_model_class: type[BaseAutoModel] = AutoModelForTextEncoding,
        device: torch.device | None = None,
        freeze_model: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Template for a Text Transformer model.

        Args:
            model_name (str): The pretrained model to use.
            model_kwargs (dict | None, optional): kwargs to pass to the model.
                Defaults to None.
            tokenizer_kwargs (dict | None, optional): kwargs to pass to the tokenizer.
                Defaults to None.
            auto_model_class (type[BaseAutoModel], optional): AutoModel class to use for
                loading the model (e.g. AutoModelForSequenceClassification).
                Defaults to AutoModelForTextEncoding.
            device (torch.device | None, optional): Torch device to use. If None, will
                select GPU if possible, else CPU. Defaults to None.
            freeze_model (bool, optional): Whether to freeze the pretrained model
                (e.g. for fine-tuning). Defaults to False.
            *args, **kwargs: To pass to nn.Module.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, **(tokenizer_kwargs or {})
        )
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            auto_model_class=auto_model_class,
            device=device,
            freeze_model=freeze_model,
            *args,
            **kwargs,
        )

    def tokenize(
        self,
        samples: str | list[str],
        batch_size: int = 1024,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenize samples using the respective model's tokenizer.

        Args:
            samples (str | list[str]): Sample(s) to tokenize.
            batch_size (int, optional): Batch size. Defaults to 1024.
            max_length (int | None, optional): Max length to pad.
                Defaults to None (max length of model).

        Returns:
            dict[str,torch.Tensor]: input_ids and attention_mask.
        """
        if isinstance(samples, str):
            samples = [samples]

        input_ids = []
        attention_mask = []

        for batch in tqdm(
            batched(samples, batch_size),
            desc="Tokenize samples",
            unit="batch",
            total=math.ceil(len(samples) / batch_size),
        ):
            encoded_dict = self.tokenizer(
                batch,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids.append(encoded_dict["input_ids"])
            attention_mask.append(encoded_dict["attention_mask"])

        return {
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attention_mask, dim=0),
        }


class SequenceClassifier(TextTransformer):

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        auto_model_class: type[BaseAutoModel] = AutoModelForSequenceClassification,
        device: torch.device | None = None,
        freeze_model: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Template for a Sequence Classifier model.

        Args:
            model_name (str): The pretrained model to use.
            model_kwargs (dict | None, optional): kwargs to pass to the model.
                Defaults to None.
            tokenizer_kwargs (dict | None, optional): kwargs to pass to the tokenizer.
                Defaults to None.
            auto_model_class (type[BaseAutoModel], optional): AutoModel class to use for
                loading the model (e.g. AutoModelForSequenceClassification).
                Defaults to AutoModelForSequenceClassification.
            device (torch.device | None, optional): Torch device to use. If None, will
                select GPU if possible, else CPU. Defaults to None.
            freeze_model (bool, optional): Whether to freeze the pretrained model
                (e.g. for fine-tuning). Defaults to False.
            *args, **kwargs: To pass to nn.Module.
        """
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            auto_model_class=auto_model_class,
            device=device,
            freeze_model=freeze_model,
            *args,
            **kwargs,
        )

    def infer(self, *args, **kwargs) -> SequenceClassifierOutput:
        return super().infer(*args, **kwargs)

    def predict(self, *args, **kwargs) -> torch.Tensor:
        """Model prediction. Applies a softmax layer after inference to predict a class.

        Returns:
            torch.Tensor: _description_
        """
        return self.infer(*args, **kwargs).logits.softmax(-1)

    def get_model_inputs(
        self,
        samples: str | list[str],
        *labels: Sequence[Hashable],
        tokenize_batch_size: int = 1024,
        max_length: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], tuple[bidict[int, Hashable], ...]]:
        """Get model inputs for sequence classification.

        Args:
            samples (str | list[str]): Samples to tokenize.
            *labels (Sequence[Hashable]): One sequence of labels per task.
            tokenize_batch_size (int, optional): Batch size for tokenization.
                Defaults to 1024.
            max_length (int | None, optional): Max length to pad.
                Defaults to None (max length of model).
        Returns:
            tuple[dict[str, torch.Tensor], tuple[bidict[int, Hashable], ...]]: Model
                inputs and a tuple of one bidict index<>label for each label sequence.
        """
        input_ids_masks = self.tokenize(
            samples=samples,
            batch_size=tokenize_batch_size,
            max_length=max_length,
        )
        index_to_label, converted_labels = convert_labels(*labels)

        return input_ids_masks | {"labels": converted_labels}, index_to_label

    def eval_epoch(
        self,
        data: DataLoader,
        data_keys: Sequence[str],
        forward_kwargs: dict | None = None,
        eval_type: str = "evaluation",
        epoch: int = 0,
    ) -> None:
        """Evaluates one epoch using loss as metric.

        Args:
            data (DataLoader): The data to evaluate on.
            data_keys (Sequence[str]): Keys of each data Tensor that the dataloader yields
                to pass it to forward call with.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.
            eval_type (str, optional): The type of evaluation (for logging).
                Defaults to "evaluation".
            epoch (int, optional): Epoch number for tensorboard. Defaults to 0.
        """
        self.eval()

        f1_total = 0
        mcc_total = 0

        for batch_idx, batch in enumerate(
            tqdm(
                data,
                desc=f"Batch {eval_type}",
                unit="batch",
                position=1,
                leave=False,
            )
        ):

            prediction = (
                self.predict(
                    **dict(zip(data_keys, batch)),
                    **(forward_kwargs or {}),
                )
                .argmax()
                .cpu()
            )
            true_labels = batch[data_keys.index("labels")]

            f1 = f1_score(true_labels, prediction)
            f1_total += f1

            mcc = matthews_corrcoef(true_labels, prediction)
            mcc_total += mcc

            # visualize with tensorboard
            self.tensorboard_writer.add_scalar(
                f"{eval_type} f1", f1, epoch * len(data) + batch_idx
            )
            self.tensorboard_writer.add_scalar(
                f"{eval_type} MCC", mcc, epoch * len(data) + batch_idx
            )

        logger.info(f"average {eval_type} f1 score: {round(f1_total/len(data),2)}")
        logger.info(f"average {eval_type} MCC: {round(mcc_total/len(data),2)}")


OutputHeadName = TypeVar("OutputHeadName", bound=str)
OutputHeadKey = TypeVar("OutputHeadKey", bound=str)


class MultiLabelSequenceClassifier(SequenceClassifier, Generic[OutputHeadName]):

    @dataclass
    class MultiLabelOutput(ModelOutput, Generic[OutputHeadKey]):
        """Dataclass for multi-label output."""

        loss: torch.FloatTensor | None = None
        losses: dict[OutputHeadKey, torch.FloatTensor] | None = None
        logits: dict[OutputHeadKey, torch.FloatTensor] | None = None
        hidden_states: tuple[torch.FloatTensor, ...] | None = None
        attentions: tuple[torch.FloatTensor, ...] | None = None

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        device: torch.device | None = None,
        freeze_model: bool = False,
        heads: dict[OutputHeadName, int] | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Multi-label classifier.

        Args:
            model_name (str): The pretrained model to use.
            model_kwargs (dict | None, optional): kwargs to pass to the model.
                Defaults to None.
            tokenizer_kwargs (dict | None, optional): kwargs to pass to the tokenizer.
                Defaults to None.
            device (torch.device | None, optional): Torch device to use. If None, will
                select GPU if possible, else CPU. Defaults to None.
            freeze_model (bool, optional): Whether to freeze the pretrained model
                (e.g. for fine-tuning). Defaults to False.
            heads (dict[OutputHeadName, int] | None, optional): Output heads to add.
                Keys will be identifiers in output, values the size of the respective heads.
                Defaults to None.
            *args, **kwargs: To pass to nn.Module.
        """
        if heads is None or len(heads) < 2:
            raise ValueError("Need at least two heads for MultiLabel classification.")
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            auto_model_class=AutoModel,
            device=device,
            freeze_model=freeze_model,
            *args,
            **kwargs,
        )

        self.dropout = nn.Dropout(0.1)

        self.heads = heads

        if heads:
            # set heads and corresponding loss functions
            for head_name, head_size in heads.items():
                setattr(
                    self,
                    f"{head_name}_head",
                    nn.Linear(self.model.config.hidden_size, head_size),
                )
                setattr(self, f"{head_name}_loss_fn", nn.CrossEntropyLoss())

        self.to(self.device)

    def forward(self, *args, **kwargs) -> MultiLabelOutput[OutputHeadName]:

        labels = kwargs.pop("labels").to(self.device)
        output = super().forward(*args, **kwargs)

        hidden_states = output.hidden_states
        attentions = output.attentions
        pooled_output = output.pooler_output  # CLS token representation
        del output  # to avoid OOME

        pooled_output = self.dropout(pooled_output)

        # get logits from heads
        logits = {
            head: getattr(self, f"{head}_head")(pooled_output) for head in self.heads
        }
        del pooled_output  # to avoid OOME

        # get losses
        losses = {
            head: getattr(self, f"{head}_loss_fn")(logit, task_labels)
            for (head, logit), task_labels in zip(
                logits.items(), labels.transpose(0, 1)
            )
        }

        return self.MultiLabelOutput(
            loss=sum(losses.values()),
            losses=losses,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def infer(self, *args, **kwargs) -> MultiLabelOutput:
        return super().infer(*args, **kwargs)

    # Inference function to get probabilities
    def predict(self, *args, **kwargs) -> dict[OutputHeadName, torch.Tensor]:
        output = self.infer(*args, **kwargs)

        # Apply softmax to get probabilities
        return {
            head_name: logit.softmax(-1) for head_name, logit in output.logits.items()
        }

    def eval_epoch(
        self,
        data: DataLoader,
        data_keys: Sequence[str],
        forward_kwargs: dict | None = None,
        eval_type: str = "evaluation",
        epoch: int = 0,
    ) -> None:
        """Evaluates one epoch using loss as metric.

        Args:
            data (DataLoader): The data to evaluate on.
            data_keys (Sequence[str]): Keys of each data Tensor that the dataloader yields
                to pass it to forward call with.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.
            eval_type (str, optional): The type of evaluation (for logging).
                Defaults to "evaluation".
            epoch (int, optional): Epoch number for tensorboard. Defaults to 0.
        """
        self.eval()

        f1_total = 0

        for batch_idx, batch in enumerate(
            tqdm(
                data,
                desc=f"Batch {eval_type}",
                unit="batch",
                position=1,
                leave=False,
            )
        ):

            prediction = self.predict(
                **dict(zip(data_keys, batch)),
                **(forward_kwargs or {}),
            )

            # apply argmax and stack
            prediction = torch.stack(
                tuple(t.argmax(1) for t in prediction.values()), 1
            ).cpu()

            # get true labels
            true_labels = batch[data_keys.index("labels")]

            # create one hot tensors
            prediction_one_hot = one_hot_multitask(
                prediction, list(self.heads.values())
            )
            true_labels_one_hot = one_hot_multitask(
                true_labels, list(self.heads.values())
            )

            f1 = f1_score(
                true_labels_one_hot,
                prediction_one_hot,
                average="weighted",
                zero_division=0,
            )
            f1_total += f1

            # visualize with tensorboard
            self.tensorboard_writer.add_scalar(
                f"{eval_type} f1", f1, epoch * len(data) + batch_idx
            )

        logger.info(f"average {eval_type} f1 score: {round(f1_total/len(data),2)}")
