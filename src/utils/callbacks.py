from argparse import Namespace
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar

__all__ = ["get_callbacks_for_given_model"]

VALIDATION_METRIC = dict(
    pretrain="val_mae_loss",
    base="val_g_loss",
    c2f="val_g_loss",
)


def get_callbacks_for_given_model(args: Namespace):
    assert args.model in ("pretrain", "base", "c2f"), f"cannot handle unknown model {args.model}"
    callbacks = []
    callbacks += [_get_checkpoint_callback(args.model)]
    # callbacks += [_get_early_stop_callback(args.model)]
    callbacks += [TQDMProgressBar(refresh_rate=100)]
    return callbacks


def _get_checkpoint_callback(model: str) -> ModelCheckpoint:
    # Depending on the mode, the val_loss has an other name
    val_loss = VALIDATION_METRIC[model]
    filename = model + "-{epoch}-{" + val_loss + ":.4f}"
    callback = ModelCheckpoint(
        monitor=val_loss,
        save_last=True,
        filename=filename,
        every_n_epochs=5,  # plays hand in hand with early stopping callback (patience)
    )
    return callback


def _get_early_stop_callback(model: str) -> EarlyStopping:
    callback = EarlyStopping(
        monitor=VALIDATION_METRIC[model],
        patience=3,
    )
    return callback
