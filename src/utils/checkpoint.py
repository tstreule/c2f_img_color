import warnings
from pathlib import Path
import torch
from torch.nn import Module


__all__ = ["set_checkpoint_vals", "get_checkpoint_path", "save_model", "load_model"]


def set_checkpoint_vals(checkpoint=None):
    cp_name = None
    cp_after_each = 20
    cp_overwrite = True

    if isinstance(checkpoint, str):
        cp_name = checkpoint
    elif isinstance(checkpoint, (tuple, list)):
        cp_name = str(checkpoint[0])
        cp_after_each, cp_overwrite = int(checkpoint[1]), bool(checkpoint[2])
    else:
        checkpoint = False

    return checkpoint, cp_name, cp_after_each, cp_overwrite


def get_checkpoint_path(name: str) -> str:
    path = "checkpoints/" + str(name).strip(".pt") + ".pt"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Module, name: str, overwrite=True):
    if overwrite:
        torch.save(model.state_dict(), get_checkpoint_path(name))
    else:
        warnings.warn(f"Model {name} not saved. Overwriting prohibited.")


def load_model(model: Module, name: str, device: torch.device = None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(get_checkpoint_path(name), map_location=device)
    model.load_state_dict(state_dict)
