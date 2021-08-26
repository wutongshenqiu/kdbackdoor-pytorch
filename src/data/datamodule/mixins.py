from abc import ABC

from torch.utils.data import Dataset, DataLoader


class PoisonDataModuleMixin(ABC):
    test_poison_dataset: Dataset

    def test_poison_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_poison_dataset,
            shuffle=self._test_shuffle,
            drop_last=self._test_drop_last,
            num_workers=self._num_workers,
            batch_size=self._batch_size
        )
