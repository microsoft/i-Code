import sys

from pytorch_lightning.callbacks.progress import ProgressBar
from tqdm import tqdm


class CustomProgressBar(ProgressBar):
    def _init_tqdm(self, desc: str, leave=False) -> tqdm:
        return tqdm(
            desc=f'{desc} ({self.process_position})',
            position=self.process_position,
            leave=leave,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    def init_sanity_tqdm(self) -> tqdm:
        return self._init_tqdm('Validation sanity check')

    def init_train_tqdm(self) -> tqdm:
        return self._init_tqdm('Training')

    def init_validation_tqdm(self) -> tqdm:
        return self._init_tqdm('Validating')

    def init_test_tqdm(self) -> tqdm:
        return self._init_tqdm('Testing', leave=True)

    def on_epoch_start(self, trainer, pl_module):
        super().on_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(f'Epoch {trainer.current_epoch}' f' ({self.process_position})')
