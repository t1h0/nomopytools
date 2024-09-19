import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import (
    AutoModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass as BaseAutoModel
from transformers.utils.generic import ModelOutput
from tqdm import tqdm
from collections.abc import Sequence
from pathlib import Path
import os
from .utils import train_validation_test_split, Device, get_datetime, to_device
from .containers import Metric

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
        device: torch.device | None = None,
        freeze_model: bool = False,
        random_seed: int | None = False,
        *args,
        **kwargs,
    ) -> None:
        """Template for a Transformer model.

        Args:
            model_name (str): The pretrained model to use.
            model_kwargs (dict | None, optional): kwargs to pass to the model.
                Defaults to None.
            auto_model_class (type[BaseAutoModel], optional): AutoModel class
                to use for loading the model (e.g. AutoModelForSequenceClassification).
                Defaults to AutoModel.
            device (torch.device | None, optional): Torch device to use. If None, will
                select GPU if possible, else CPU. Defaults to None.
            freeze_model (bool, optional): Whether to freeze the pretrained model
                (e.g. for fine-tuning). Defaults to False.
            random_seed (int, optional). Seed to use for randomization. If None, will
                let torch decide. Defaults to None.
            *args, **kwargs: To pass to nn.Module.
        """
        super().__init__(*args, **kwargs)

        self.model = auto_model_class.from_pretrained(
            model_name, **(model_kwargs or {})
        )

        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        self.device = device or Device
        self.to(self.device)

        self.random_seed = random_seed

        # for tensorboard visualization
        self.tensorboard_writer = SummaryWriter()

    def forward(self, *args, **kwargs) -> ModelOutput:
        # to make things easier, we'll always return a ModelOutput
        kwargs["return_dict"] = True

        return self.model(
            *to_device(args, self.device), **to_device(kwargs, self.device)
        )

    def infer(self, *args, **kwargs) -> ModelOutput:
        """Model inference. Forward passes args and kwargs without gradient computation.

        Returns:
            ModelOutput: The output.
        """
        self.eval()

        with torch.no_grad():
            return self(*args, **kwargs)

    def perform_training(
        self,
        model_input: dict[str, torch.Tensor],
        forward_kwargs: dict | None = None,
        batch_size: int = 32,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        epochs: int = 4,
        export_checkpoints: bool = False,
        export_complete: bool | str | Path = False,
        *args,
        **kwargs,
    ) -> None:
        """Trains the classifier.

        Args:
            model_input (dict[str,torch.Tensor]): Model inputs that will be
                train-validation-test split and passed to forward call with the
                respective keys from the dictionary. E.g. input_ids and attention_mask.
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
            export_checkpoints (bool, optional): Whether to export a general checkpoint
                of the model after each epoch to
                export/{ClassName}/general-checkpoints/{yyyy}-{mm}-{dd}_{hh}-{mm}-{ss}-{ms}_Epoch-{epoch}.tar
                Defaults to False.
            export_complete (bool | str | Path, optional): Whether export the
                model' state_dict after training is complete to
                export/{ClassName}/torchscripts/{yyyy}-{mm}-{dd}_{hh}-{mm}-{ss}-{ms}.pt.
                If a string or a Path is given, the model is exported to that location.
                Defaults to False.
        """
        # get dataloaders
        data = train_validation_test_split(
            *model_input.values(),
            batch_size=batch_size,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_seed=self.random_seed,
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

        # get forward_keys
        model_input_keys = list(model_input.keys())

        # train

        for epoch in tqdm(
            range(epochs), desc="Epochs", unit="epoch", position=0, leave=True
        ):
            # iterate over epochs
            logger.info(f"=== Epoch {epoch} ===")

            for step in ("train", "validate", "test"):
                # iterate over training, validation and test
                # get data
                data_step = getattr(data, step)

                for batch_idx, batch in enumerate(
                    tqdm(
                        data_step,
                        position=1,
                        desc=f"Batch {step}",
                        unit="batch",
                        leave=False,
                    )
                ):
                    # iterate over batches

                    if step == "train":
                        # set to train mode
                        self.train()

                        metrics = self.train_epoch(
                            batch,
                            model_input_keys,
                            optimizer,
                            lr_scheduler,
                            forward_kwargs,
                        )

                    else:
                        # set to eval mode
                        self.eval()

                        metrics = getattr(self, f"{step}_epoch")(
                            batch, model_input_keys, forward_kwargs
                        )

                    # execute and get metric
                    # logger.info(f"Write {step} metric(s) for batch {batch_idx} to tensorboard..")
                    for metric in metrics:
                        # log to tensorboard
                        self.tensorboard_writer.add_scalar(
                            f"{step} {metric.name}",
                            metric.value,
                            epoch * len(data_step) + batch_idx,
                        )

                if step == "train":
                    if export_checkpoints:
                        logger.info("Export general checkpoint..")
                        self.export_general_checkpoint(
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            loss=next(m.value for m in metrics if m.name == "loss"),
                            epoch=epoch,
                            path=None,
                        )
                    if epoch == epochs - 1 and export_complete:
                        logger.info("Export model..")
                        self.export_state_dict(
                            export_complete
                            if isinstance(export_complete, (str, Path))
                            and export_complete
                            else None
                        )

        logger.info("Training complete!")

    def train_epoch(
        self,
        batch: DataLoader,
        data_keys: Sequence[str],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        forward_kwargs: dict | None = None,
    ) -> tuple[Metric, ...]:
        """Trains one epoch.

        Args:
            batch (DataLoader): The batch to train on.
            data_keys (Sequence[str]): Keys of each data Tensor that the dataloader yields
                to pass it to forward call with.
            optimizer (Optimizer): The optimizer to use.
            lr_scheduler (LRScheduler): The learning rate scheduler to use.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.

        Returns:
            tuple[Metric,...]: The loss.
        """
        # clear any previously calculated gradients
        self.zero_grad()

        # forward pass
        output = self(
            **dict(zip(data_keys, batch)),
            **(forward_kwargs or {}),
        )

        # get loss
        out = output.loss.item()

        # backward pass
        output.loss.backward()

        # Clip norm of the gradients to 1.0 to prevent
        # exploding gradients problem
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # update learning rate
        lr_scheduler.step()

        return (Metric("loss", out),)

    def validate_epoch(
        self,
        batch: DataLoader,
        data_keys: Sequence[str],
        forward_kwargs: dict | None = None,
    ) -> tuple[Metric, ...]:
        """Validates one epoch.

        Args:
            batch (DataLoader): The batch to train on.
            data_keys (Sequence[str]): Keys of each data Tensor that the dataloader yields
                to pass it to forward call with.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.

        Returns:
            tuple[Metric,...]: The loss.
        """
        return self.eval_epoch(
            batch=batch,
            data_keys=data_keys,
            forward_kwargs=forward_kwargs,
        )

    def test_epoch(
        self,
        batch: DataLoader,
        data_keys: Sequence[str],
        forward_kwargs: dict | None = None,
    ) -> tuple[Metric, ...]:
        """Tests one epoch.

        Args:
            batch (DataLoader): The batch to train on.
            data_keys (Sequence[str]): Keys of each data Tensor that the dataloader yields
                to pass it to forward call with.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.

        Returns:
            tuple[Metric,...]: The loss.
        """
        return self.eval_epoch(
            batch=batch,
            data_keys=data_keys,
            forward_kwargs=forward_kwargs,
        )

    def eval_epoch(
        self,
        batch: DataLoader,
        data_keys: Sequence[str],
        forward_kwargs: dict | None = None,
    ) -> tuple[Metric, ...]:
        """Evaluates one epoch using loss as metric.

        Args:
            batch (DataLoader): The batch to train on.
            data_keys (Sequence[str]): Keys of each data Tensor that the dataloader yields
                to pass it to forward call with.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.

        Returns:
            tuple[Metric,...]: The loss.
        """
        with torch.no_grad():
            output = self(
                **dict(zip(data_keys, batch)),
                **(forward_kwargs or {}),
            )

        return (Metric("loss", output.loss.item()),)

    def export_state_dict(self, path: str | Path | None = None) -> None:
        """Export the current state dict.

        Args:
            path (str | Path | None, optional): Path to export to. If None, will export
                to export/{ClassName}/state-dicts/{yyyy}-{mm}-{dd}_{hh}-{mm}-{ss}-{ms}.pt.
                Defaults to None.
        """
        if path is None:
            dirs = f"export/{self.__class__.__name__}/state-dicts/"
            os.makedirs(dirs, exist_ok=True)
            path = f"{dirs}{get_datetime()}.pt"
        torch.save(self.state_dict(), path)

    def export_general_checkpoint(
        self,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        loss: torch.Tensor,
        epoch: int,
        path: str | Path | None = None,
    ) -> None:
        """Export a general checkpoint (for Inference and/or Resuming Training).

        Args:
            optimizer (Optimizer): The optimizer used during training.
            lr_scheduler (LRScheduler): The learning rate scheduler used during training.
            loss (torch.Tensor): The last loss.
            epoch (int): The last epoch.
            path (str | Path): The path to export to. Should be a .tar file.
                If None, will export to
                export/{ClassName}/general-checkpoints/{yyyy}-{mm}-{dd}_{hh}-{mm}-{ss}-{ms}_Epoch-{epoch}.tar
                Defaults to None.
        """
        if path is None:
            dirs = f"export/{self.__class__.__name__}/general-checkpoints/"
            os.makedirs(dirs, exist_ok=True)
            path = f"{dirs}{get_datetime()}_Epoch-{epoch}.tar"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "loss": loss,
            },
            path,
        )
