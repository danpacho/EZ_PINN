from dataclasses import asdict, dataclass
import json
from typing import Union
import torch
import torch.nn as nn

from abc import ABCMeta, abstractmethod

from src.train.model_component import ModelComponent


@dataclass
class TrainConfig:
    """
    TrainConfig class to store the training configuration.
    """

    lr: float
    """
    Learning rate.

    > Note: Batch Size ∝ Learning rate 
    """

    batch_size: int
    """
    Batch size.
    > Note: Batch Size ∝ Learning rate
    >
    > Note: Mini-batch = gonna be a subset of the dataset
    """

    epoch: int
    """
    How many times the model will see the entire dataset.
    """

    early_stopping: Union[False, int]
    """
    Early stopping tolerance.
    
    If model is not improving for `early_stopping` epochs, stop the training.

    default: 30
    """

    def __post_init__(self) -> None:
        """
        Post init method to validate the TrainConfig.
        """
        if self.lr <= 0:
            raise ValueError("Learning rate should be greater than 0.")

        if self.batch_size <= 0:
            raise ValueError("Batch size should be greater than 0.")

        if self.epoch <= 0:
            raise ValueError("Epoch should be greater than 0.")

        if self.early_stopping is None:
            BASE_TOLERANCE = 30
            self.early_stopping = BASE_TOLERANCE
            return

        if self.early_stopping is False:
            self.early_stopping = 0
            return

        if self.early_stopping is not None and self.early_stopping <= 0:
            raise ValueError("Early stopping should be greater than 0.")

    def to_json(self) -> str:
        """
        Convert the dataclass instance to a JSON string.
        """
        return json.dumps(asdict(self))


class Trainer(metaclass=ABCMeta):
    def __init__(
        self, train_config: TrainConfig, model_component: ModelComponent
    ) -> None:
        """
        Initialize Trainer.

        Args:
            train_config (TrainConfig): Train configuration.
        """

        self.train_config = train_config
        self.model_component = model_component

        self.optimizer = self.define_optimizer()
        self.criterion = self.define_criterion()
        self.scheduler = self.define_scheduler(self.optimizer)

        self.device: str = None
        self._dependency_injected: bool = False

    @property
    def model(self) -> nn.Module:
        return self.model_component.model

    def inject_dependencies(
        self,
        device: str,
    ) -> None:
        """
        Inject trainer dependencies.

        Device

        Args:
            device (str): Device name
            model (nn.Module): Model(torch)
        """
        self.device = device
        self._dependency_injected = True

    @property
    def trainer_seed(self) -> str:
        """
        Seed string for trainer

        Returns:
            str: Seed string
        """
        optimizer_config = self.optimizer.__str__().replace("\n", "")
        criterion_config = self.criterion.__str__().replace("\n", "")
        lr_scheduler_config = self.scheduler.__dict__.__str__().replace("\n", "")
        train_config = self.train_config.__str__().replace("\n", "")

        return f"{optimizer_config}_{criterion_config}_{lr_scheduler_config}_{train_config}"

    @abstractmethod
    def define_optimizer(self) -> torch.optim.Optimizer:
        """
        Define optimizer.
        """
        raise NotImplementedError

    @abstractmethod
    def define_criterion(self) -> nn.Module:
        """
        Define criterion.
        """
        raise NotImplementedError

    @abstractmethod
    def define_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Define scheduler.
        """
        raise NotImplementedError

    @property
    def latest_lr(self) -> float:
        """
        Get the latest learning rate.
        """
        return self.scheduler.get_last_lr()[0]

    @abstractmethod
    def compute_accuracy(self, *args, **kwargs) -> None:
        """
        Compute performance metrics for the model.
        """
        raise NotImplementedError

    def train(self, train_data: any) -> tuple[float, float, float]:
        """
        Train the model.

        Args:
            train_data (any): DataLoader enumerated object

            ```python
            for step, train_data in enumerate(train_data):
                trainer.train(train_data)
            ```
            So you have to unpack the `train_data` in the `train` method.

        Returns:
            tuple: `(train_loss, train_accuracy, lr)`
        """
        self._train_setup()
        train_loss, train_accuracy = self._train(train_data)
        lr = self.latest_lr
        return train_loss, train_accuracy, lr

    @abstractmethod
    def _train(self, train_data: any) -> tuple[float, float]:
        """
        Train the model.

        Args:
            train_data (any): DataLoader enumerated object

            ```python
            for step, train_data in enumerate(train_data):
                trainer.train(train_data)
            ```
            So you have to unpack the `train_data` in the `train` method.

        Returns:
            tuple: `(train_loss, train_accuracy)`
        """
        raise NotImplementedError

    def _train_setup(self) -> None:
        self.model.train()
        self.optimizer.zero_grad()

    def evaluate(self, validation_data: any) -> tuple[float, float]:
        """
        Validate the model.

        Args:
            validation_data (any): DataLoader enumerated object

            ```python
            for step, val_data in enumerate(validation_data):
                trainer.evaluate(val_data)
            ```
            So you have to unpack the `val_data` in the `evaluate` method.

        Returns:
            tuple: `(val_loss, val_accuracy)`
        """
        self._evaluate_setup()
        return self._evaluate(validation_data)

    def _evaluate_setup(self) -> None:
        self.model.eval()

    @abstractmethod
    def _evaluate(self, validation_data: any) -> tuple[float, float]:
        """
        Validate the model.

        Args:
            validation_data (any): DataLoader enumerated object

            ```python
            for step, val_data in enumerate(validation_data):
                trainer.evaluate(val_data)
            ```
            So you have to unpack the `val_data` in the `evaluate` method.

        Returns:
            tuple: `(val_loss, val_accuracy)`
        """
        raise NotImplementedError
