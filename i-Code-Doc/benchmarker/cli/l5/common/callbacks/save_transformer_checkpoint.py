from pathlib import Path
from typing import Union
import pytorch_lightning as pl

class SaveTransformerCheckpoint(pl.Callback):
    def __init__(self, save_path: Union[str, Path]):
        self.save_path = save_path