from tqdm import tqdm
from transformers.trainer_callback import ProgressCallback


class MyProgressCallback(ProgressCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(total=state.max_steps)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(self,
                           args,
                           state,
                           control,
                           eval_dataloader=None,
                           **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(total=len(eval_dataloader),
                                           leave=self.training_bar is None)
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            _ = logs.pop('total_flos', None)
            self.training_bar.write(str(logs))

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.close()
            self.training_bar = None
