# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import rootutils
from beartype.typing import Tuple
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

@utils.task_wrapper
def sample(cfg: DictConfig) -> Tuple[dict, dict]:
    """Samples given checkpoint on a datamodule prediction set.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random

    print (f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    print (f"Instantiating model <{cfg.model._target_}>")
    # model: LightningModule = hydra.utils.instantiate(
    #     cfg.model,
    #     model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
    #     diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
    #     data_cfg=cfg.data.data_cfg,
    #     path_cfg=cfg.paths,
    #     inference_cfg=cfg.inference,
    # )
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    print ("Loading checkpoint!")
    model = model.load_from_checkpoint(
        cfg.ckpt_path
    )

    print (f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "model": model,
        "trainer": trainer,
    }

    trainer.predict(model=model, datamodule=datamodule)
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    sample(cfg)


if __name__ == "__main__":
    # register_custom_omegaconf_resolvers()
    main()