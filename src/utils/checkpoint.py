import warnings
from pathlib import Path
import torch
from torch.nn import Module


__all__ = ["set_cp_args", "secure_cp_path", "save_model", "load_model"]


def set_cp_args(checkpoint=None):
    """
    Returns:
            checkpoint_args (tuple):
            - (str) checkpoint path
            - (int) checkpoint after each
            - (bool) overwrite checkpoint if already exists
    """
    if isinstance(checkpoint, str):
        checkpoint = (checkpoint, 20, True)

    path, after_each, overwrite = [fc(cp) for cp, fc in zip(checkpoint, [str, int, bool])]

    return path, after_each, overwrite


def secure_cp_path(name: str) -> str:
    path = str(name).strip(".pt") + ".pt"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Module, name: str, overwrite=True):
    if overwrite:
        torch.save(model.state_dict(), name)
    else:
        warnings.warn(f"Model {name} not saved. Overwriting prohibited.")


def load_model(model: Module, name: str, device: torch.device = None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(name, map_location=device)
    model.load_state_dict(state_dict)
