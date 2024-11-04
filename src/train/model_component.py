import torch
import torch.nn as nn
from torchsummary import summary


class InvalidPathError(Exception):
    def __init__(self, path: str) -> None:
        self.path = path
        self.message = f"Invalid path, should be `*.pth`.\nCheck given path {path}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class ModelComponent:
    def __init__(self, name: str, model: nn.Module) -> None:
        self.name = name
        self.model = model

    def summary(self, input_size: tuple[int, int, int]) -> None:
        summary(self.model, input_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output(Predicted) tensor
        """
        return self.model(X)

    def _check_pth(self, path: str) -> bool:
        is_valid_pth = path.endswith(".pth")
        return is_valid_pth

    def load_model(self, path: str) -> None:
        """
        Load model params

        Args:
            path (str): `*.pth` file path for torch
        """
        if not self._check_pth(path):
            raise InvalidPathError(path)

        self.model.load_state_dict(torch.load(path))

    def save_model(self, path: str) -> None:
        """
        Save model params

        Args:
            path (str): `*.pth` file path for torch
        """
        if not self._check_pth(path):
            raise InvalidPathError(path)

        torch.save(self.model.state_dict(), path)

    def __str__(self) -> str:
        return f"Model: {self.name}"

    @property
    def model_seed(self) -> str:
        hyperparams = self.model.__str__().replace("\n", "")
        model_name = self.name

        return f"{model_name}_{hyperparams}"
