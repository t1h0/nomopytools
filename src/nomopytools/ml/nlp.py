import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTextEncoding,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm
from bidict import bidict
from collections.abc import Sequence, Hashable
import math
from .utils import batched, convert_labels
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
        """
        self.eval()

        for batch in tqdm(
            data,
            desc=f"Batch {eval_type}",
            unit="batch",
            position=1,
            leave=False,
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
            mcc = matthews_corrcoef(true_labels, prediction)

            logger.info(f"test f1 score: {f1}")
            logger.info(f"test MCC: {mcc}")
