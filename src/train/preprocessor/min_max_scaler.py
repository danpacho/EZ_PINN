import torch


class MinMaxScaler:
    """
    MinMaxScaler
    X' = ( X - min(X) ) / ( max(X) - min(X) )
    X = X' * ( max(X) - min(X) ) + min(X)
    """

    def __init__(self, data: torch.Tensor, epsilon=1e-10):
        self.min = data.min()
        self.max = data.max()
        self.epsilon = epsilon  # Add small constant to avoid division by zero

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform data to [0, 1] range
        """
        return (data - self.min) / (self.max - self.min + self.epsilon)

    def recover(self, data: torch.Tensor) -> torch.Tensor:
        """
        Recover transformed data to original scale
        """
        return data * (self.max - self.min + self.epsilon) + self.min

    def fit(self, data: torch.Tensor):
        """
        Fit MinMaxScaler with new data
        """
        self.min = data.min()
        self.max = data.max()


# Test the modified scaler
if __name__ == "__main__":

    def test_min_max_scaler():
        data = torch.tensor([1, 2000, 0.0000003, 4, 5], dtype=torch.float64)

        scaler = MinMaxScaler(data)
        scaled_data = scaler.transform(data)
        recovered_data = scaler.recover(scaled_data)

        print("Original data:", data)
        print("Scaled data:", scaled_data)
        print("Recovered data:", recovered_data)

    test_min_max_scaler()
