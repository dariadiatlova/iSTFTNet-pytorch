import os
import shutil

from pathlib import Path


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config_path: str, ckpt_path: str) -> None:
    """ Copies config to the checkpoint directory """
    config_name = Path(config_path).name
    target_path = Path(ckpt_path).joinpath(config_name)
    if config_path != target_path:
        os.makedirs(ckpt_path, exist_ok=True)
        shutil.copyfile(config_path, Path(ckpt_path).joinpath(config_name))
