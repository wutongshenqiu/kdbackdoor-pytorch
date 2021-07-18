from pathlib import PurePath

from pydantic import BaseModel


class BaseConfig(BaseModel):
    project_dir_path: PurePath = PurePath(__file__).parent.parent
    checkpoints_dir_path: PurePath = project_dir_path / "checkpoints"
    logs_dir_path: PurePath = project_dir_path / "lightening_logs"

    class Config:
        # allow for `PurePath` type
        arbitrary_types_allowed = True


base_config = BaseConfig()
