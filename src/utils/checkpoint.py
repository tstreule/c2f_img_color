import warnings
from pathlib import Path
import torch
from torch.nn import Module


__all__ = ["set_checkpoint_args", "get_checkpoint_path", "save_model", "load_model"]


def set_checkpoint_args(checkpoint=None):
    """
    Returns:
        A tuple `(make_checkpoints, checkpoint_args)`.
        make_checkpoints (bool): Whether to make checkpoints
        checkpoint_args (tuple):
            - (str) checkpoint path
            - (int) checkpoint after each
            - (bool) overwrite checkpoint if already exists
    """
    if isinstance(checkpoint, str):
        checkpoint = (checkpoint, 20, True)

    make_checkpoint = False if checkpoint[0] is None else True

    checkpoint_args = [fc(cp) for cp, fc in zip(checkpoint, [str, int, bool])]
    return make_checkpoint, checkpoint_args


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
