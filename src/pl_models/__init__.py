from typing import Type

from .kdbackdoor import KDBackdoorModel
from .normal import NormalModel
from .filetune import FinetuneModel

from pytorch_lightning import LightningDataModule

_AVALIABLE_MODEL_TYPE = {
    "normal": NormalModel,
    "kdbackdoor": KDBackdoorModel,
    "finetune": FinetuneModel
}


def get_model_type(model_name: str) -> Type[LightningDataModule]:
    if (model_type := _AVALIABLE_MODEL_TYPE.get(model_name)) is not None:
        return model_type
    else:
        raise ValueError(f"model `{model_name}` is not supported!")
