"""Global variables."""
from typing import Literal, get_args
from pathlib import Path

Phases = Literal["train","validate","test"]
PhasesArgs = get_args(Phases)

def _get_export_path(exporting_instance:object) -> Path:
    return Path("export") / exporting_instance.__class__.__name__

def get_checkpoint_export_path(exporting_instance: object) -> Path:
    return _get_export_path(exporting_instance) / "checkpoints"

def get_state_dicts_export_path(exporting_instance: object) -> Path:
    return _get_export_path(exporting_instance) / "state-dicts"