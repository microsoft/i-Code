from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .custom_progress_bar import CustomProgressBar
from .save_prediction_callback import SavePredictionCallback
from .save_transformer_checkpoint import SaveTransformerCheckpoint

__all__ = [
    'get_checkpoint_callback',
    'get_early_stopping_callback',
    'CustomProgressBar',
    'SavePredictionCallback',
    'SaveTransformerCheckpoint',
]


def get_checkpoint_callback(output_dir, metric, save_top_k=1, lower_is_better=False):
    """Saves the best model by validation ROUGE2 score."""
    exp = f"{{val_{metric}:.4f}}-{{step_count}}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=exp,
        monitor=f"val_{metric}",
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
        save_last=True,
    )
    return checkpoint_callback


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=f"val_{metric}",  # does this need avg?
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )
