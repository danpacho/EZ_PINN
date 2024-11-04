from typing import Union
import torch


from src.train.sampler.abstract_sampler import AbstractSampler


class Sampler1D(AbstractSampler):
    """
    1D samplers.
    """

    def __init__(self, ranges: tuple[float, float], num_samples: int):
        self.ranges = ranges
        self.num_samples = num_samples

    def grid_sample(self, num_samples: Union[int, None] = None) -> torch.Tensor:
        sample_grid_1d = torch.linspace(
            self.ranges[0],
            self.ranges[1],
            self.num_samples if num_samples is None else num_samples,
        )
        return sample_grid_1d

    def random_sample(self, num_samples: Union[int, None] = None) -> torch.Tensor:
        range_start: float = self.ranges[0]
        range_length: float = self.ranges[1] - self.ranges[0]
        random_sample = torch.rand(
            self.num_samples if num_samples is None else num_samples
        )
        sample_1d_random = random_sample * range_length + range_start

        return sample_1d_random

    @property
    def range_len(self) -> int:
        return self.ranges[1] - self.ranges[0]
