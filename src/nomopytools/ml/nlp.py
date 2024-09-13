import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTextEncoding,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm
from collections.abc import Sequence, Hashable
from typing import Literal
from .containers import ForwardSplitInput, DataSplit
from .utils import convert_labels
from .transformer import Transformer, BaseAutoModel, logger


class TextTransformer(Transformer):

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        auto_model_class: type[BaseAutoModel] = AutoModelForTextEncoding,
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
            *args, **kwargs: To pass to nn.Module.
        """
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            auto_model_class=auto_model_class,
            *args,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, **(tokenizer_kwargs or {})
        )

    def tokenize(
        self, samples: str | Sequence[str], max_length: int | None = None
    ) -> tuple[ForwardSplitInput, ForwardSplitInput]:
        """Tokenize samples using the respective model's tokenizer.

        Args:
            samples (str | Sequence[str]): Sample(s) to tokenize.
            max_length (int | None, optional): Max length to pad.
                Defaults to None (max length of model).

        Returns:
            tuple[ForwardSplitInput,ForwardSplitInput]: input_ids and attention_mask as
                ForwardSplitInput objects.
        """
        if isinstance(samples, str):
            samples = [samples]

        input_ids = []
        attention_mask = []

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
            attention_mask.append(encoded_dict["attention_mask"])

        return (
            ForwardSplitInput("input_ids", torch.cat(input_ids, dim=0)),
            ForwardSplitInput("attention_mask", torch.cat(attention_mask, dim=0)),
        )

    def perform_training(
        self,
        samples: Sequence[str],
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
            samples (Sequence[str]): Samples (untokenized).
            forward_inputs (ForwardSplitInput | Sequence[ForwardSplitInput]):
                Additional ForwardSplitInput tuple(s) that will be train-validation-test
                 split and passed to forward call.
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

        Raises:
            ValueError: _description_
        """
        if isinstance(forward_inputs, ForwardSplitInput):
            forward_inputs = (forward_inputs,)

        # tokenize
        input_ids_mask = self.tokenize(samples)

        return super().perform_training(
            forward_inputs=(*input_ids_mask, *forward_inputs),
            forward_kwargs=forward_kwargs,
            batch_size=batch_size,
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
            *args, **kwargs: To pass to nn.Module.
        """
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            auto_model_class=auto_model_class,
            *args,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, **(tokenizer_kwargs or {})
        )

    def predict(self, *args, **kwargs) -> torch.Tensor:
        return self.infer(*args, **kwargs).logits.argmax(1)

    def perform_training(
        self,
        samples: Sequence[str],
        *labels: Sequence[Hashable],
        forward_inputs: ForwardSplitInput | Sequence[ForwardSplitInput] | None = None,
        forward_kwargs: dict | None = None,
        batch_size: int = 32,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        epochs: int = 4,
    ) -> None:
        """Trains the classifier.

        Args:
            samples (Sequence[str]): Samples (untokenized).
            *labels (Sequence[Hashable]): One sequence of labels per task.
            forward_inputs (ForwardSplitInput | Sequence[ForwardSplitInput] | None,
                optional): Additional ForwardSplitInput tuple(s) that will be
                train-validation-test split and passed to forward call.
                Defaults to None.
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

        Raises:
            ValueError: _description_
        """
        if len({len(task_labels) for task_labels in labels}) != 1 or len(
            labels[0]
        ) != len(samples):
            raise ValueError(
                "Each sequence of labels must have as many labels as there are samples."
            )

        if isinstance(forward_inputs, ForwardSplitInput):
            forward_inputs = (forward_inputs,)
        elif forward_inputs is None:
            forward_inputs = ()

        # convert labels
        input_labels = ForwardSplitInput("labels", convert_labels(*labels))

        return super().perform_training(
            samples=samples,
            forward_inputs=(input_labels, *forward_inputs),
            forward_kwargs=forward_kwargs,
            batch_size=batch_size,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            epochs=epochs,
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

        if eval_type == "validation":
            return super().eval_epoch(eval_type, data, data_keys, forward_kwargs)

        for batch in tqdm(
            data.test,
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

