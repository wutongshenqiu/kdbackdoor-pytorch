from typing import Any, Iterator, Iterable, Tuple, Optional, List, Mapping
import os

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parameter import Parameter
from torch import Tensor
import torch.nn.functional as F

from src.networks import get_network, get_backdoor
from src.data import get_datamodule, BaseDataModule
from .config.kdbackdoor import Config
from .utils import (
    get_loss_function,
    evaluate_benign_accuracy,
    evaluate_backdoor_success_rate,
    compute_accuracy
)

_config = Config()


class KDBackdoorModel(pl.LightningModule):
    name: str = "kdbackdoor"

    def __init__(
        self, *,
        teacher_network: str = _config.teacher_network,
        pretrain_teacher_path: Optional[str] = None,
        student_network: str = _config.student_network,
        loss_function: str = _config.loss_function,
        target_label: int = _config.target_label,
        lr_teacher: float = _config.lr_teacher,
        lr_student: float = _config.lr_student,
        lr_backdoor: float = _config.lr_backdoor,
        max_epochs: int = _config.max_epochs,
        momentum: float = _config.momentum,
        epoch_boundries: List[int] = _config.epoch_boundries,
        poison_rate: float = _config.poison_rate,
        temperature: int = _config.temperature,
        backdoor_l2_factor: float = _config.backdoor_l2_factor,
        alpha: float = _config.alpha,
        clip_min: float = _config.clip_min,
        clip_max: float = _config.clip_max,
        datamodule_name: str = _config.datamodule_name,
        **datamodule_kwargs
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self._datamodule = get_datamodule(datamodule_name, **datamodule_kwargs)

        self._backdoor_network = get_backdoor(
            shape=self._datamodule.shape,
            mean=self._datamodule.mean,
            std=self._datamodule.std,
            clip_min=clip_min,
            clip_max=clip_max
        )
        self._teacher_network = get_network(teacher_network)
        # TODO
        # not clear if this will conflict with `resume_from_checkpoint`
        if pretrain_teacher_path and os.path.exists(pretrain_teacher_path):
            self._teacher_network.load_state_dict(
                torch.load(pretrain_teacher_path)
            )
        self._student_network = get_network(student_network)

        self._loss_function = get_loss_function(loss_function)

        # this can be only used when `drop_last` = True
        assert self._datamodule._train_drop_last and self._datamodule._test_drop_last
        target_label_tensor = self._make_target_tensor(
            self._datamodule._batch_size, target_label, torch.int64
        )
        self.register_buffer("_target_label_tensor", target_label_tensor)

    def configure_optimizers(self) -> Any:
        """configure optimizers for `knowledge distillation backdoor`

        Three optimizers will be created:
            - optimizers for teacher model
                - MultiStepLR
                    - instead of `PiecewiseConstantDecay` in original tensorflow implementation
                - SGD
                - momentum = 0.9
            - optimizers for student model
                - MultiStepLR
                    - instead of `PiecewiseConstantDecay` in original tensorflow implementation
                - SGD
                - momentum = 0.9
            - optimizers for backdoor
                - constant lr(1e-4)
        """
        def _make_sgd_optimizer(
            params: Iterator[Parameter],
            lr: float,
            momentum: float = self.hparams.momentum
        ) -> SGD:
            return SGD(
                params=params, lr=lr, momentum=momentum
            )

        def _make_multistep_scheduler(
            optimizer: Optimizer,
            milestones: Iterable[int] = self.hparams.epoch_boundries,
            gamma: float = 0.1
        ) -> MultiStepLR:
            return MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=gamma
            )

        teacher_optimizer = _make_sgd_optimizer(
            params=self._teacher_network.parameters(),
            lr=self.hparams.lr_teacher,
        )

        student_optimizer = _make_sgd_optimizer(
            params=self._student_network.parameters(),
            lr=self.hparams.lr_student,
        )

        backdoor_optimizer = SGD(
            params=self._backdoor_network.parameters(),
            lr=self.hparams.lr_backdoor,
        )

        teacher_scheduler = _make_multistep_scheduler(teacher_optimizer)
        student_scheduler = _make_multistep_scheduler(student_optimizer)

        # HACK
        # this scheduler has no effect for `backdoor learning rate`
        # but to keep training sequence(backdoor -> teacher -> student) as original code
        backdoor_scheduler = _make_multistep_scheduler(
            optimizer=backdoor_optimizer)

        return [backdoor_optimizer, teacher_optimizer, student_optimizer], \
            [backdoor_scheduler, teacher_scheduler, student_scheduler]

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        optimizer_idx: int
    ) -> STEP_OUTPUT:
        """perform anti-distillation backdoor attack

        this process contains three steps:
            1. training backdoor
            2. training teahcer model
            3. training student model
        """
        x, y = batch

        if optimizer_idx == 0:
            # here we inject trigger after doing transforms
            backdoor_x = self._backdoor_network(x)
            loss = self._train_backdoor(
                backdoor_x=backdoor_x,
            )
            self._epoch_log_dict({
                "backdoor_loss": loss
            })

        if optimizer_idx == 1:
            backdoor_x = self._backdoor_network(x)
            loss = self._train_teacher(
                x=x,
                backdoor_x=backdoor_x,
                y=y,
            )
            self._epoch_log_dict({
                "teacher_loss": loss
            })

        if optimizer_idx == 2:
            loss = self._train_student(
                x=x,
                y=y
            )
            self._epoch_log_dict({
                "student_loss": loss
            })

        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        student_test_acc = evaluate_benign_accuracy(
            model=self._student_network,
            dataloader=self.datamodule.test_dataloader(),
            device=self.device
        )
        teacher_test_acc = evaluate_benign_accuracy(
            model=self._teacher_network,
            dataloader=self.datamodule.test_dataloader(),
            device=self.device
        )
        self.log_dict({
            "student_test_acc": student_test_acc,
            "teacher_test_acc": teacher_test_acc
        }, on_step=False, on_epoch=True)

        self.log_dict({
            "student_backdoor_success_rate": evaluate_backdoor_success_rate(
                model=self._student_network,
                backdoor=self._backdoor_network,
                dataloader=self.datamodule.test_dataloader(),
                target_label=self.hparams.target_label,
                device=self.device
            ),
            "teacher_backdoor_success_rate": evaluate_backdoor_success_rate(
                model=self._teacher_network,
                backdoor=self._backdoor_network,
                dataloader=self.datamodule.test_dataloader(),
                target_label=self.hparams.target_label,
                device=self.device
            )
        }, on_step=False, on_epoch=True)

    def _train_backdoor(
        self, *,
        backdoor_x: Tensor,
    ) -> Tensor:
        logits_from_teacher = self._teacher_network(backdoor_x)
        logits_from_student = self._student_network(backdoor_x)

        teacher_loss = self._loss_function(
            logits_from_teacher, self._target_label_tensor
        )
        student_loss = self._loss_function(
            logits_from_student, self._target_label_tensor
        )
        trigger_l2_penalty = self._l2_norm_without_sqrt(
            self._backdoor_network.mask * self._backdoor_network.trigger
        )
        self._epoch_log_dict({
            "backdoor_teacher_loss": teacher_loss,
            "backdoor_student_loss": student_loss,
            "backdoor_trigger_l2_penalty": trigger_l2_penalty
        })

        return teacher_loss + student_loss + \
            trigger_l2_penalty * self.hparams.backdoor_l2_factor

    def _train_teacher(
        self, *,
        x: Tensor,
        backdoor_x: Tensor,
        y: Tensor,
    ) -> Tensor:
        T = self.hparams.temperature

        benign_logits = self._teacher_network(x)
        backdoor_logits = self._teacher_network(backdoor_x)

        benign_loss = self._loss_function(
            benign_logits / T, y
        )
        backdoor_loss = self._loss_function(
            backdoor_logits / T,
            self._target_label_tensor
        )

        self._epoch_log_dict({
            "teacher_benign_loss": benign_loss,
            "teacher_backdoor_loss": backdoor_loss
        })

        train_acc = compute_accuracy(benign_logits, y)
        self._epoch_log_dict({
            "teacher_train_acc": train_acc
        })

        return (1 - self.hparams.poison_rate) * benign_loss + \
            self.hparams.poison_rate * backdoor_loss

    def _train_student(
        self, *,
        x: Tensor,
        y: Tensor
    ) -> Tensor:
        logits_from_teacher = self._teacher_network(x)
        logits_from_student = self._student_network(x)

        soft_loss = self._calculate_soft_loss_original(
            logits_from_student=logits_from_student,
            logits_from_teacher=logits_from_teacher
        )
        hard_loss = self._loss_function(
            logits_from_student, y
        )

        self._epoch_log_dict({
            "student_soft_loss": soft_loss,
            "student_hard_loss": hard_loss
        })

        return self.hparams.alpha * soft_loss + \
            (1 - self.hparams.alpha) * hard_loss

    def _epoch_log_dict(self, dictionary: Mapping[str, Any]) -> None:
        self.log_dict(
            dictionary=dictionary,
            on_step=True,
            on_epoch=True
        )

    # TODO
    # different from orginal version
    # `softmax_cross_entropy_with_logits` version is similar to https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501/2
    # below version from https://github.com/peterliht/knowledge-distillation-pytorch/blob/ef06124d67a98abcb3a5bc9c81f7d0f1f016a7ef/model/net.py#L100
    def _calculate_soft_loss(
        self, *,
        logits_from_student: Tensor,
        logits_from_teacher: Tensor
    ) -> Tensor:
        T = self.hparams.temperature
        return F.kl_div(
            F.log_softmax(logits_from_student / T, dim=1),
            F.softmax(logits_from_teacher / T, dim=1),
            reduction="batchmean"
        ) * T ** 2

    # same version as `softmax_cross_entropy_with_logits`
    def _calculate_soft_loss_original(
        self, *,
        logits_from_student: Tensor,
        logits_from_teacher: Tensor
    ) -> Tensor:
        T = self.hparams.temperature

        soft_label_from_teacher = F.softmax(logits_from_teacher / T, dim=1)
        return self._softmax_cross_entropy_with_logits(
            logits=logits_from_student / T,
            labels=soft_label_from_teacher
        )

    @staticmethod
    def _softmax_cross_entropy_with_logits(
        logits: Tensor,
        labels: Tensor
    ) -> Tensor:
        log_probs = F.log_softmax(logits, dim=1)

        return -(labels * log_probs).sum() / logits.shape[0]

    @staticmethod
    def _l2_norm_without_sqrt(x: Tensor) -> Tensor:
        return torch.sum(x ** 2) / 2

    @staticmethod
    def _make_target_tensor(
        length: int,
        value: int,
        dtype: torch.dtype
    ) -> Tensor:
        t = torch.empty(length, dtype=dtype).fill_(value)

        return t

    @property
    def datamodule(self) -> BaseDataModule:
        return self._datamodule
