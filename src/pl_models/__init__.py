from typing import Type

from .kdbackdoor import KDBackdoorModel
from .normal import NormalModel
from .kdfiletune import KDFinetuneModel
from .badnet import BadNetModel
from .finetune import FinetuneModel

from pytorch_lightning import LightningDataModule

_AVALIABLE_MODEL_TYPE = {
    "normal": NormalModel,
    "kdbackdoor": KDBackdoorModel,
    "kdfinetune": KDFinetuneModel,
    "finetune": FinetuneModel,
    "badnet": BadNetModel
}


def get_model_type(model_name: str) -> Type[LightningDataModule]:
    if (model_type := _AVALIABLE_MODEL_TYPE.get(model_name)) is not None:
        return model_type
    else:
        raise ValueError(f"model `{model_name}` is not supported!")
