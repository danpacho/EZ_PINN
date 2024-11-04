import torch
import torch.nn as nn


class LRP:
    """
    Layer-wise Relevance Propagation (LRP)

    Ref: https://git.tu-berlin.de/gmontavon/lrp-tutorial
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.device = "cpu"

        self.model.to(self.device)
        self.model.eval()

        self.layers: list[nn.Module] = []
        self.forward_activations: list[torch.Tensor] = []
        self.relevance_score: list[torch.Tensor] = []

        self._extract_linear_layers(model)
        self._forward_calc = False

    @property
    def layer_count(self) -> int:
        """
        Get the number of linear layers in the model

        Returns:
            int: layer count
        """
        return len(self.layers)

    def _extract_linear_layers(self, model: nn.Module):
        linear_layers = []

        def _collect_linear_layers(layer):
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)
            elif isinstance(layer, nn.Sequential):
                for sub_layer in layer.children():
                    _collect_linear_layers(sub_layer)
            elif isinstance(layer, nn.Module):
                for sub_layer in layer.children():
                    _collect_linear_layers(sub_layer)

        _collect_linear_layers(model)
        self.layers = linear_layers

    def forward(self, input_tensor: torch.Tensor) -> None:
        """
        Forward propagation

        Note: **it should be called before backpropagation**

        Args:
            input_tensor (torch.Tensor): target input tensor
        """
        input_tensor = input_tensor.to(self.device)

        self.forward_activations = []
        self.forward_activations.append(input_tensor)
        # Removed print statements for clarity

        activation = input_tensor
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                activation = layer(activation)
                if i < self.layer_count - 1:
                    activation = torch.relu(activation)
                self.forward_activations.append(activation)

        self._forward_calc = True

    def back_propagation_epsilon(self) -> None:
        """
        Backpropagation of relevance scores using `epsilon` rule

        Raises:
            RuntimeError: if forward calculation is not done before backpropagation
        """
        if not self._forward_calc:
            raise RuntimeError(
                "Forward calculation must be done before back_propagation."
            )

        # Initialize relevance scores
        self.relevance_score = [None] * (self.layer_count + 1)
        R_L = self.forward_activations[-1]  # Last activation (output)
        self.relevance_score[self.layer_count] = R_L  # Initial relevance

        # Loop over layers from L-1 down to 0
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layers[i]

            # Get activations
            A_i = self.forward_activations[i]
            # Ensure A_i is 2D (batch_size, features)
            if A_i.dim() == 1:
                A_i = A_i.unsqueeze(0)

            # Get weights and biases
            W = layer.weight  # Shape: [out_features, in_features]
            B = layer.bias  # Shape: [out_features]

            # Apply rho to weights and biases
            def rho(w):
                return w

            w = rho(W)
            b = rho(B)

            # Compute Z = A_i @ w.T + b
            Z = torch.matmul(A_i, w.t()) + b.unsqueeze(0)

            # Apply incr to Z
            def incr(z):
                epsilon = 1e-6  # Small positive number to avoid division by zero
                return z + epsilon * torch.sign(z) + 1e-9

            Z = incr(Z)

            # Get R_{i+1}
            R_i_plus_1 = self.relevance_score[i + 1]
            # Ensure R_i_plus_1 is 2D
            if R_i_plus_1.dim() == 1:
                R_i_plus_1 = R_i_plus_1.unsqueeze(0)

            # Compute S = R_{i+1} / Z
            S = R_i_plus_1 / Z

            # Handle potential division by zero
            S = torch.where(Z != 0, S, torch.zeros_like(S))

            # Compute C = S @ w
            C = torch.matmul(S, w)

            # Compute R_i = A_i * C
            R_i = A_i * C

            # Store R_i
            self.relevance_score[i] = R_i

        self._forward_calc = False  # Reset for next forward pass

    def back_propagation_z_beta(self) -> None:
        """
        Backpropagation of relevance scores using `Z-beta` rule

        Raises:
            RuntimeError: if forward calculation is not done before backpropagation
        """

        if not self._forward_calc:
            raise RuntimeError(
                "Forward calculation must be done before backpropagation."
            )

        # Initialize relevance scores
        self.relevance_score = [None] * (self.layer_count + 1)
        R_L = self.forward_activations[-1]  # Last activation (output)
        self.relevance_score[self.layer_count] = R_L  # Initial relevance

        # Loop over layers from L-1 down to 0
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layers[i]

            # Get activations
            A_i = self.forward_activations[i]
            # Ensure A_i is 2D (batch_size, features)
            if A_i.dim() == 1:
                A_i = A_i.unsqueeze(0)

            # Get weights and biases
            W = layer.weight  # Shape: [out_features, in_features]
            B = layer.bias  # Shape: [out_features]

            # Get R_{i+1}
            R_i_plus_1 = self.relevance_score[i + 1]
            # Ensure R_i_plus_1 is 2D
            if R_i_plus_1.dim() == 1:
                R_i_plus_1 = R_i_plus_1.unsqueeze(0)

            # **Z-beta Rule Implementation at Input Layer**
            w = W
            wp = torch.clamp(w, min=0)  # w+
            wm = torch.clamp(w, max=0)  # w-

            # **Set lower and upper bounds (adjust according to your data)**
            # Assuming input features are normalized between -1 and 1
            lower_bound = -1.0
            upper_bound = 1.0

            lb = torch.full_like(A_i, fill_value=lower_bound)
            hb = torch.full_like(A_i, fill_value=upper_bound)

            # **Compute z according to z-beta rule**
            z = (
                torch.matmul(A_i, w.t())
                - torch.matmul(lb, wp.t())
                - torch.matmul(hb, wm.t())
                + 1e-9
            )  # [batch_size, out_features]

            # Compute s
            s = R_i_plus_1 / z  # Shape: [batch_size, out_features]

            # Handle potential division by zero
            s = torch.where(z != 0, s, torch.zeros_like(s))

            # Compute c, cp, cm
            c = torch.matmul(s, w)  # [batch_size, in_features]
            cp = torch.matmul(s, wp)  # [batch_size, in_features]
            cm = torch.matmul(s, wm)  # [batch_size, in_features]

            # Compute R_i
            R_i = A_i * c - lb * cp - hb * cm  # [batch_size, in_features]

            # Store R_i
            self.relevance_score[i] = R_i

        self._forward_calc = False  # Reset for next forward pass

    def back_propagation(self) -> None:
        """
        Backpropagation of relevance scores

        Raises:
            RuntimeError: if forward calculation is not done before backpropagation
        """
        if not self._forward_calc:
            raise RuntimeError(
                "Forward calculation must be done before backpropagation."
            )

        # Initialize relevance scores
        self.relevance_score = [None] * (self.layer_count + 1)
        R_L = self.forward_activations[-1]  # Last activation (output)
        self.relevance_score[self.layer_count] = R_L  # Initial relevance

        # Loop over layers from L-1 down to 0
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layers[i]

            # Get activations
            A_i = self.forward_activations[i]
            # Ensure A_i is 2D (batch_size, features)
            if A_i.dim() == 1:
                A_i = A_i.unsqueeze(0)

            # Get weights and biases
            W = layer.weight  # Shape: [out_features, in_features]
            B = layer.bias  # Shape: [out_features]

            # Get R_{i+1}
            R_i_plus_1 = self.relevance_score[i + 1]
            # Ensure R_i_plus_1 is 2D
            if R_i_plus_1.dim() == 1:
                R_i_plus_1 = R_i_plus_1.unsqueeze(0)

            if i == 0:
                # **Z-beta Rule Implementation at Input Layer**
                w = W
                wp = torch.clamp(w, min=0)  # w+
                wm = torch.clamp(w, max=0)  # w-

                # **Set lower and upper bounds (adjust according to your data)**
                # Assuming input features are normalized between -1 and 1
                lower_bound = -1.0
                upper_bound = 1.0

                lb = torch.full_like(A_i, fill_value=lower_bound)
                hb = torch.full_like(A_i, fill_value=upper_bound)

                # **Compute z according to z-beta rule**
                z = (
                    torch.matmul(A_i, w.t())
                    - torch.matmul(lb, wp.t())
                    - torch.matmul(hb, wm.t())
                    + 1e-9
                )  # [batch_size, out_features]

                # Compute s
                s = R_i_plus_1 / z  # Shape: [batch_size, out_features]

                # Handle potential division by zero
                s = torch.where(z != 0, s, torch.zeros_like(s))

                # Compute c, cp, cm
                c = torch.matmul(s, w)  # [batch_size, in_features]
                cp = torch.matmul(s, wp)  # [batch_size, in_features]
                cm = torch.matmul(s, wm)  # [batch_size, in_features]

                # Compute R_i
                R_i = A_i * c - lb * cp - hb * cm  # [batch_size, in_features]

                # Store R_i
                self.relevance_score[i] = R_i
            else:
                # **Epsilon Rule for Other Layers**
                # Apply rho to weights and biases (identity function here)
                def rho(w):
                    return w

                w = rho(W)
                b = rho(B)

                # Compute Z = A_i @ w.T + b
                Z = torch.matmul(A_i, w.t()) + b.unsqueeze(0)

                # Apply incr to Z
                def incr(z):
                    epsilon = 1e-6  # Small positive number to avoid division by zero
                    return z + epsilon * torch.sign(z) + 1e-9

                Z = incr(Z)

                # Compute S = R_{i+1} / Z
                S = R_i_plus_1 / Z

                # Handle potential division by zero
                S = torch.where(Z != 0, S, torch.zeros_like(S))

                # Compute C = S @ w
                C = torch.matmul(S, w)

                # Compute R_i = A_i * C
                R_i = A_i * C

                # Store R_i
                self.relevance_score[i] = R_i

        self._forward_calc = False  # Reset for next forward pass


if __name__ == "__main__":
    # Example usage
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.Linear(2, 2),
    )

    lrp = LRP(model)

    input_tensor = torch.rand(2)
    lrp.forward(input_tensor)
    lrp.back_propagation()
    print(lrp.relevance_score[0])
