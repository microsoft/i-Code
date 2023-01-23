from argparse import Namespace
from typing import Any, Dict, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class BetterMlFlowLogger(pl.loggers.MLFlowLogger):
    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for k, v in params.items():
            if len(str(v)) >= 250:
                v = str(v)[:249]
            try:
                self.experiment.log_param(self.run_id, k, v)
            except:
                print(f"Could not log {k}: {v} ({len(str(v))})")
