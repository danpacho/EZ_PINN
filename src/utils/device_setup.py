from typing import Literal, Union
import torch


def get_device() -> Union[Literal["mps"], Literal["cpu"]]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return device


if __name__ == "__main__":
    print(get_device())
