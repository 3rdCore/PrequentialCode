import wandb
from pytorch_lightning.loggers import WandbLogger


class BufferedWandbLogger(WandbLogger):
    def __init__(self, flush_every_n_steps: int = 50, verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.flush_every_n_steps = flush_every_n_steps
        self.buffer = []  # Stores (metrics, step) tuples
        self.step_counter = 0
        self.verbose = verbose

    def log_metrics(self, metrics, step=None):
        # Add metrics and step to buffer

        # Store each log entry (metrics, step) in the buffer
        self.buffer.append((metrics, step))

        self.step_counter += 1

        # Flush the buffer after `flush_interval` steps
        if len(self.buffer) >= self.flush_every_n_steps:
            self._flush_metrics()

    def _flush_metrics(self):
        if self.buffer:
            self.verbose and print(f"{self.__class__.__name__}: Flushing logs... count: {self.step_counter}.")
            # Flush each buffered entry in the order it was added
            for metrics, step in self.buffer:
                super().log_metrics(metrics, step=step)
            # Clear buffer after flushing
            self.buffer.clear()

    def on_epoch_end(self):
        # Ensure to flush any remaining metrics at the end of the epoch
        self._flush_metrics()

    def on_train_end(self):
        # Ensure everything is flushed when training ends
        self._flush_metrics()
