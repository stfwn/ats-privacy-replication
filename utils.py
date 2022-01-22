from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            trainer.save_checkpoint(
                Path(self.dirpath) / f"epoch={trainer.current_epoch}.ckpt"
            )


def split_augmentations(aug_list: str) -> list:
    aug_list = aug_list.split("+")
    aug_list = [l.split("-") for l in aug_list]
    aug_list = [[int(i) for i in l if i] for l in aug_list]
    return aug_list
