import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTextEncoding,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm
from collections.abc import Sequence, Hashable
import math
from .utils import convert_labels, batched
from .transformer import Transformer, BaseAutoModel, logger


class TextTransformer(Transformer):

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        auto_model_class: type[BaseAutoModel] = AutoModelForTextEncoding,
        device: torch.device | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Template for a Text Transformer model.

        Args:
            model_name (str): The model to use.
            model_kwargs (dict | None, optional): kwargs to pass to the model.
                Defaults to None.
            tokenizer_kwargs (dict | None, optional): kwargs to pass to the tokenizer.
                Defaults to None.
            auto_model_class (type[BaseAutoModel], optional): AutoModel class to use for
                loading the model (e.g. AutoModelForSequenceClassification).
                Defaults to AutoModelForTextEncoding.
            device (torch.device | None, optional): Torch device to use. If None, will
                select GPU if possible, else CPU. Defaults to None.
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

    def perform_training(
        self,
        samples: list[str] | None = None,
        model_input: dict[str, torch.Tensor] | None = None,
        forward_kwargs: dict | None = None,
        tokenization_batch_size: int = 1024,
        train_batch_size: int = 32,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        epochs: int = 4,
        *args,
        **kwargs,
    ) -> None:
        """Trains the classifier.

        Args:
            samples (list[str] | None, optional): Samples (untokenized). Defaults to None.
            forward_inputs (dict[str,torch.Tensor] | None, optional): Additional input
                tensors that will be train-validation-test split and passed to
                forward call with the respective keys from the dictionary.
                Defaults to None.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.
            tokenization_batch_size (int, optional): Batch size for tokenization.
                Defaults to 1024.
            train_batch_size (int, optional): Batch size for training. Defaults to 32.
            train_size (float, optional): Train set size (as proportion).
                Defaults to 0.8.
            val_size (float, optional): Validation set size (as proportion).
                Defaults to 0.1.
            test_size (float, optional): Test set size (as proportion).
                Defaults to 0.1.
            epochs (int, optional): Number of epochs to run. Defaults to 4.

        Raises:
            ValueError: _description_
        """
        if model_input is None:
            model_input = {}

        if samples is not None:
            # tokenize
            model_input = (
                self.tokenize(samples, batch_size=tokenization_batch_size) | model_input
            )

        return super().perform_training(
            model_input=model_input,
            forward_kwargs=forward_kwargs,
            batch_size=train_batch_size,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            epochs=epochs,
        )


class SequenceClassifier(TextTransformer):

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        auto_model_class: type[BaseAutoModel] = AutoModelForSequenceClassification,
        device: torch.device | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Template for a Sequence Classifier model.

        Args:
            model_name (str): The model to use.
            model_kwargs (dict | None, optional): kwargs to pass to the model.
                Defaults to None.
            tokenizer_kwargs (dict | None, optional): kwargs to pass to the tokenizer.
                Defaults to None.
            auto_model_class (type[BaseAutoModel], optional): AutoModel class to use for
                loading the model (e.g. AutoModelForSequenceClassification).
                Defaults to AutoModelForSequenceClassification.
            device (torch.device | None, optional): Torch device to use. If None, will
                select GPU if possible, else CPU. Defaults to None.
            *args, **kwargs: To pass to nn.Module.
        """
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            auto_model_class=auto_model_class,
            device=device,
            *args,
            **kwargs,
        )

    def predict(self, *args, **kwargs) -> torch.Tensor:
        return self.infer(*args, **kwargs).logits.argmax(1)

    def perform_training(
        self,
        samples: list[str] | None = None,
        *labels: Sequence[Hashable],
        model_input: dict[str, torch.Tensor] | None = None,
        forward_kwargs: dict | None = None,
        tokenization_batch_size: int = 1024,
        train_batch_size: int = 32,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        epochs: int = 4,
    ) -> None:
        """Trains the classifier.

        Args:
            samples (list[str] | None, optional): Samples (untokenized). Defaults to None.
            *labels (Sequence[Hashable]): One sequence of labels per task.
            forward_inputs (dict[str,torch.Tensor] | None, optional): Additional input
                tensors that will be train-validation-test split and passed to
                forward call with the respective keys from the dictionary.
                Defaults to None.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.
            tokenization_batch_size (int, optional): Batch size for tokenization.
                Defaults to 1024.
            train_batch_size (int, optional): Batch size for training. Defaults to 32.
            train_size (float, optional): Train set size (as proportion).
                Defaults to 0.8.
            val_size (float, optional): Validation set size (as proportion).
                Defaults to 0.1.
            test_size (float, optional): Test set size (as proportion).
                Defaults to 0.1.
            epochs (int, optional): Number of epochs to run. Defaults to 4.

        Raises:
            ValueError: If labels per task are not equal on length or are not matching
                number of samples.
        """
        if model_input is None:
            model_input = {}

        if labels:
            if samples is not None and (
                len({len(task_labels) for task_labels in labels}) != 1
                or len(labels[0]) != len(samples)
            ):
                raise ValueError(
                    "Each sequence of labels must have as many labels as there are samples."
                )
            # convert labels
            model_input = {"labels": convert_labels(*labels)} | model_input

        return super().perform_training(
            samples=samples,
            model_input=model_input,
            forward_kwargs=forward_kwargs,
            tokenization_batch_size=tokenization_batch_size,
            train_batch_size=train_batch_size,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            epochs=epochs,
        )

    def test_epoch(
        self,
        data: DataLoader,
        data_keys: Sequence[str],
        forward_kwargs: dict | None = None,
    ) -> None:
        """Tests one epoch.

        Args:
            data (DataLoader): The data to test on.
            data_keys (Sequence[str]): Keys of each data Tensor that the dataloader yields
                to pass it to forward call with.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.
        """
        for batch in tqdm(
            data,
            desc="Batch test",
            unit="batch",
            position=1,
            leave=False,
        ):

            prediction = (
                self.predict(
                    **dict(zip(data_keys, batch)),
                    **(forward_kwargs or {}),
                )
                .detach()
                .cpu()
            )
            true_labels = batch[data_keys.index("labels")]

            f1 = f1_score(true_labels, prediction)
            mcc = matthews_corrcoef(true_labels, prediction)

            logger.info(f"test f1 score: {f1}")
            logger.info(f"test MCC: {mcc}")
