from typing import Optional

from torch.utils.data import Dataset

from .base import BaseDataModule
from ..dataset import PartialDataset


# TODO
# class variable?
class PartialDataModule(BaseDataModule):

    def __init__(
        self, *,
        datamodule: BaseDataModule,
        train_partial_rate: float,
        test_partial_rate: float,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # HACK
        if not hasattr(datamodule, "_train_dataset"):
            datamodule.setup()
        self._original_datamodule = datamodule
        self._train_partial_rate = train_partial_rate
        self._test_partial_rate = test_partial_rate

    def setup(self, stage: Optional[str] = None) -> None:
        # HACK
        self._original_datamodule._train_shuffle = False
        self._original_datamodule._test_shuffle = False
        self._original_datamodule._train_drop_last = False
        self._original_datamodule._test_drop_last = False
        
        self._train_dataset = PartialDataset(
            dataloader=self._original_datamodule.train_dataloader(),
            partial_rate=self._train_partial_rate
        )
        self._test_dataset = PartialDataset(
            dataloader=self._original_datamodule.test_dataloader(),
            partial_rate=self._test_partial_rate
        )
        # HACK
        # to save memory
        del self._original_datamodule

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset
