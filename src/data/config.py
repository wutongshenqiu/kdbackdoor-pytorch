from pathlib import PurePath

from pydantic import BaseModel


class Settings(BaseModel):
    root_dir: PurePath = PurePath("~/workspace/dataset")

    train_shuffle: bool = True
    train_drop_last: bool = True

    test_shuffle: bool = False
    test_drop_last: bool = True

    # TODO
    # should be GPU * 4
    num_workers: int = 4
    batch_size: int = 128

    class Config:
        # allow for `PurePath` type
        arbitrary_types_allowed = True


settings = Settings()
