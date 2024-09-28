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
import time
import os
import sys
import glob
from itertools import islice
from dataclasses import dataclass
from .utils import train_validation_test_split, Device, get_datetime, to_device
from .containers import Metric, DataSplit, Checkpoint
from .globals import (
    Phases,
    PhasesArgs,
    get_checkpoint_export_path,
    get_state_dicts_export_path,
)

# logger setup
from loguru import logger

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class Transformer(nn.Module):

    @dataclass
    class _ModelState:
        """Stores the current model state AFTER having performed batch of epoch of phase."""

        optimizer_state_dict: dict | None = None
        lr_scheduler_state_dict: dict | None = None
        epoch: int | None = 0
        phase: Phases | None = None
        batch: int | None = None

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
        logger.info(f"Using device {self.device}")
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

    def start_training(
        self,
        model_input: dict[str, torch.Tensor],
        forward_kwargs: dict | None = None,
        batch_size: int = 32,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        epochs: int = 4,
        export_checkpoints: int | None = None,
        export_complete: bool | str | Path = False,
    ) -> None:
        """Start training.

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
            export_checkpoints (int | None, optional): Interval in minutes in which
                to export a checkpoint of the model to
                export/{ClassName}/checkpoints/{yyyy}-{mm}-{dd}_{hh}-{mm}-{ss}-{ms}.tar
                In None, won't export checkpoints. Defaults to None.
            export_complete (bool | str | Path, optional): Whether export the
                model' state_dict after training is complete to
                export/{ClassName}/state_dicts/{yyyy}-{mm}-{dd}_{hh}-{mm}-{ss}-{ms}.pt.
                If a string or a Path is given, the model is exported to that location.
                Defaults to False.
        """
        # get data, optimizer, lr_scheduler
        data, optimizer, lr_scheduler = self._get_datasplit_optimizer_lrscheduler(
            model_input=model_input,
            batch_size=batch_size,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            epochs=epochs,
        )

        return self._perform_training(
            data=data,
            model_input_keys=list(model_input.keys()),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            forward_kwargs=forward_kwargs,
            epochs=epochs,
            export_checkpoints=export_checkpoints,
            export_complete=export_complete,
        )

    def resume_training(
        self,
        model_input: dict[str, torch.Tensor],
        checkpoint: Checkpoint | None = None,
        forward_kwargs: dict | None = None,
        batch_size: int = 32,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        epochs: int = 4,
        export_checkpoints: int | None = None,
        export_complete: bool | str | Path = False,
    ) -> None:
        """Resume training after loading a checkpoint.

        Args:
            model_input (dict[str,torch.Tensor]): Model inputs that will be
                train-validation-test split and passed to forward call with the
                respective keys from the dictionary. E.g. input_ids and attention_mask.
            checkpoint (Checkpoint | None, optional): Checkpoint, loaded with
                torch.load. If None, will scan export folder and let the user select.
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
            export_checkpoints (int | None, optional): Interval in minutes in which
                to export a checkpoint of the model to
                export/{ClassName}/checkpoints/{yyyy}-{mm}-{dd}_{hh}-{mm}-{ss}-{ms}.tar
                In None, won't export checkpoints. Defaults to None.
            export_complete (bool | str | Path, optional): Whether export the
                model' state_dict after training is complete to
                export/{ClassName}/state-dicts/{yyyy}-{mm}-{dd}_{hh}-{mm}-{ss}-{ms}.pt.
                If a string or a Path is given, the model is exported to that location.
                Defaults to False.
        """
        if checkpoint is None:
            checkpoint = self._load_checkpoint()

        if checkpoint["random_seed"] != self.random_seed:
            raise ValueError("Random seed of checkpoint must match that of the model.")

        # get data, optimizer, lr_scheduler
        data, optimizer, lr_scheduler = self._get_datasplit_optimizer_lrscheduler(
            model_input=model_input,
            batch_size=batch_size,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            epochs=epochs,
        )

        # load checkpoint
        self.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        return self._perform_training(
            data=data,
            model_input_keys=list(model_input.keys()),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            forward_kwargs=forward_kwargs,
            epochs=epochs,
            start_epoch=checkpoint["epoch"],
            start_phase=checkpoint["phase"],
            start_batch=checkpoint["batch"] + 1,
            export_checkpoints=export_checkpoints,
            export_complete=export_complete,
        )

    def _perform_training(
        self,
        data: DataSplit,
        model_input_keys: list[str],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        forward_kwargs: dict | None = None,
        epochs: int = 4,
        start_epoch: int = 0,
        start_phase: Phases = "train",
        start_batch: int = 0,
        export_checkpoints: int | None = None,
        export_complete: bool | str | Path = False,
        *args,
        **kwargs,
    ) -> None:
        """Trains the classifier.

        Args:
            data (DataSplit): Train, validate and test set.
            model_input_keys (list[str]): Keys to pass the Tensors that each DataLoader
                in data yields to the model forward call with.
            optimizer (Optimizer): The optimizer to use.
            lr_scheduler (LRScheduler): The learning rate scheduler to use.
            forward_kwargs (dict | None, optional): Additional kwargs to pass to
                forward call. If None will pass no additional kwargs. Defaults to None.
            epochs (int, optional): Number of epochs to run. Defaults to 4.
            start_epoch (int, optional): The epoch to start with (may be >0 if resuming
                training). Defaults to 0.
            start_batch (int, optional): The batch to start with (may be >0 if resuming
                training). Defaults to 0.
            start_phase ("train" | "validate" | "test", optional): The phase to start
                with. Defaults to "train".
            export_checkpoints (int | None, optional): Interval in minutes in which
                to export a checkpoint of the model to
                export/{ClassName}/checkpoints/{yyyy}-{mm}-{dd}_{hh}-{mm}-{ss}-{ms}.tar
                In None, won't export checkpoints. Defaults to None.
            export_complete (bool | str | Path, optional): Whether export the
                model' state_dict after training is complete to
                export/{ClassName}/state_dicts/{yyyy}-{mm}-{dd}_{hh}-{mm}-{ss}-{ms}.pt.
                If a string or a Path is given, the model is exported to that location.
                Defaults to False.
        """
        # states holds model states.
        current_state = None

        # set phases to run
        phases = PhasesArgs[PhasesArgs.index(start_phase) :]

        t0 = time.time()

        # train
        try:
            for current_epoch in tqdm(
                range(start_epoch, epochs),
                desc="Epochs",
                unit="epoch",
                position=0,
                leave=True,
            ):

                for current_phase in phases:
                    # iterate over training, validation and test

                    # get data
                    data_phase: DataLoader = getattr(data, current_phase)

                    # set start batch
                    if current_phase != start_phase:
                        start_batch = 0

                    if start_batch >= len(data_phase):
                        continue

                    if current_phase == "train":
                        # set to train mode
                        self.train()
                    else:
                        # set to eval mode
                        self.eval()

                    for current_batch, batch in zip(
                        range(start_batch, len(data_phase)),
                        tqdm(
                            islice(data_phase, start_batch, None),
                            position=1,
                            desc=f"Batch {current_phase}",
                            unit="batch",
                            leave=False,
                            total=len(data_phase) - start_batch,
                        ),
                    ):

                        # iterate over batches

                        if current_phase == "train":

                            metrics = self.train_epoch(
                                batch,
                                model_input_keys,
                                optimizer,
                                lr_scheduler,
                                forward_kwargs,
                            )

                        else:
                            metrics: tuple[Metric, ...] = getattr(
                                self, f"{current_phase}_epoch"
                            )(batch, model_input_keys, forward_kwargs)

                        # execute and get metric
                        # logger.info(f"Write {step} metric(s) for batch {batch_idx} to tensorboard..")
                        for metric in metrics:
                            # log to tensorboard
                            self.tensorboard_writer.add_scalar(
                                f"{current_phase} {metric.name}",
                                metric.value,
                                current_epoch * len(data_phase) + current_batch,
                            )

                        # save current state
                        current_state = Transformer._ModelState(
                            optimizer_state_dict=optimizer.state_dict(),
                            lr_scheduler_state_dict=lr_scheduler.state_dict(),
                            epoch=current_epoch,
                            phase=current_phase,
                            batch=current_batch,
                        )

                        if (
                            export_checkpoints is not None
                            and time.time() - t0 >= export_checkpoints * 60
                        ):
                            t0 = time.time()
                            logger.info(
                                f"Export checkpoint. Next in {export_checkpoints} minutes."
                            )
                            self.export_checkpoint(current_state)

                    if (
                        current_phase == "train"
                        and current_epoch == epochs - 1
                        and export_complete
                    ):
                        logger.info("Export model..")
                        self.export_state_dict(
                            export_complete
                            if isinstance(export_complete, (str, Path))
                            and export_complete
                            else None
                        )
        except KeyboardInterrupt as e:
            logger.info("Abort!")
            if current_state is not None:
                self.export_checkpoint(current_state)
                logger.info("Checkpoint successfully exported.")
                sys.exit(0)
            raise e

        logger.info("Training complete!")

    def _get_datasplit_optimizer_lrscheduler(
        self,
        model_input: dict[str, torch.Tensor],
        batch_size: int = 32,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        epochs: int = 4,
    ) -> tuple[DataSplit, Optimizer, LRScheduler]:
        """Split data into train, validate and test set and also define optimizer and
        learning rate scheduler.

        Args:
            model_input (dict[str,torch.Tensor]): Model inputs that will be
                train-validation-test split and passed to forward call with the
                respective keys from the dictionary. E.g. input_ids and attention_mask.
            batch_size (int, optional): Batch size. Defaults to 32.
            train_size (float, optional): Train set size (as proportion).
                Defaults to 0.8.
            val_size (float, optional): Validation set size (as proportion).
                Defaults to 0.1.
            test_size (float, optional): Test set size (as proportion).
                Defaults to 0.1.

        Returns:
            tuple[DataSplit, Optimizer, LRScheduler]: The datasets, the optimizer and
                the learning rate scheduler.
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
            num_warmup_steps=0,
            num_training_steps=len(data.train) * epochs,
        )

        return data, optimizer, lr_scheduler

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
            dirs = get_state_dicts_export_path(self)
            os.makedirs(dirs, exist_ok=True)
            path = dirs / (get_datetime() + ".pt")
        torch.save(self.state_dict(), path)

    def export_checkpoint(
        self,
        state: _ModelState,
        path: str | Path | None = None,
    ) -> None:
        """Export a checkpoint (for Inference and/or Resuming Training).

        Args:
            state (_ModelState): The state to base the checkpoint on.
            path (str | Path): The path to export to. Should be a .tar file.
                If None, will export to
                export/{ClassName}/checkpoints/{yyyy}-{mm}-{dd}_{hh}-{mm}-{ss}-{ms}.tar
                Defaults to None.
        """
        if state.optimizer_state_dict is None or state.lr_scheduler_state_dict is None:
            raise ValueError(
                "Need Optimizer and LRScheduler for exporting a checkpoint."
            )
        if path is None:
            dirs = get_checkpoint_export_path(self)
            os.makedirs(dirs, exist_ok=True)
            path = dirs / (get_datetime() + ".tar")
        torch.save(
            {
                "epoch": state.epoch,
                "phase": state.phase,
                "batch": state.batch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": state.optimizer_state_dict,
                "lr_scheduler_state_dict": state.lr_scheduler_state_dict,
                "random_seed": self.random_seed,
            },
            path,
        )

    def _load_checkpoint(self) -> Checkpoint:
        """Scans for checkpoints and loads the selected checkpoint.

        Raises:
            FileNotFoundError: If no checkpoints could be found.

        Returns:
            Checkpoint: The loaded checkpoint.
        """
        # scan for checkpoints
        checkpoint_export_path = get_checkpoint_export_path(self)
        if not os.path.exists(checkpoint_export_path) or not (
            checkpoints := sorted(glob.glob(f"{str(checkpoint_export_path)}/*.tar"))
        ):
            raise FileNotFoundError(
                "Can't find export/checkpoints folder for loading checkpoints."
            )

        # print checkpoints
        print("Available checkpoints:\n")
        print(
            "\n".join(
                f"[{checkpoint_idx}]: {os.path.basename(checkpoint)}"
                for checkpoint_idx, checkpoint in enumerate(checkpoints)
            )
        )

        # wait for user selection
        while not (
            selection := input(
                "Please select a checkpoint by submitting the respective number: "
            )
        ).isnumeric() or not 0 <= int(selection) <= len(checkpoints):
            input("Please select a valid checkpoint!")
            for _ in range(2):
                sys.stdout.write("\x1b[1A")  # Move the cursor up one line
                sys.stdout.write("\x1b[2K")  # Clear the entire line
            sys.stdout.flush()

        # return
        return torch.load(checkpoints[int(selection)], weights_only=True)
