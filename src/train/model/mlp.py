from typing import Literal, Union

import torch
import torch.nn as nn

ActivationType = Literal[
    "relu",
    "leaky_relu",
    "tanh",
    "sigmoid",
    "identity",
]


class MLP(nn.Module):
    """
    Multi-layer perceptron model
    """

    def __init__(
        self,
        model_architecture: list[int],
        activation_functions: Union[list[ActivationType], ActivationType],
        identity_last_layer: bool = True,
    ):
        """
        Generate model declaratively.

        Args:
            model_architecture (list[int]): model architecture
            activation_functions (list[str]): activation functions
        """
        super(MLP, self).__init__()

        self.identity_last_layer = identity_last_layer

        if isinstance(activation_functions, str):
            activation_functions = [activation_functions] * (
                len(model_architecture) - 1
            )

        # if len(activation_functions) != len(model_architecture) - 1:
        #     raise ValueError(
        #         f"Number of activation functions must be equal to number of {len(model_architecture) - 1}, current model architecture: {model_architecture}"
        #     )

        self.layers = nn.ModuleList()
        for i in range(len(model_architecture) - 1):
            is_last_layer = i == len(model_architecture) - 1
            curr_activation = activation_functions[i]
            self.layers.append(
                nn.Sequential(
                    nn.Linear(model_architecture[i], model_architecture[i + 1]),
                    (
                        self._get_activation(curr_activation)
                        if not (is_last_layer and identity_last_layer)
                        else nn.Identity()
                    ),
                )
            )

    def _get_activation(self, activation: ActivationType):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "identity":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, x):
        """
        Forward pass
        """
        for layer in self.layers:
            x = layer(x)

        return x

    @property
    def model_size(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def model_report(self):
        return f"Model size: {self.model_size}\nArchtecutre: {self.layers}"


if __name__ == "__main__":
    model = MLP(
        [10, 20, 30, 40],
        "relu",
        identity_last_layer=True,
    )
    print(model.model_report)
    print(model.model_size)

    model2 = MLP(
        [10, 20, 30, 40],
        ["relu", "leaky_relu", "identity"],
    )
    print(model2.model_report)
    print(model2.model_size)

    res = model2(torch.randn(10, 10))
    print(res.shape)
