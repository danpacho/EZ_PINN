from random import sample
import torch

from src.train.sampler.abstract_sampler import AbstractSampler
from src.train.sampler.sampler_1d import Sampler1D


class Sampler2D(AbstractSampler):
    """
    2D samplers.
    """

    def __init__(
        self,
        x_ranges: tuple[float, float],
        x_num_samples: int,
        y_ranges: tuple[float, float],
        y_num_samples: int,
    ):
        self.x_sampler = Sampler1D(ranges=x_ranges, num_samples=x_num_samples)
        self.y_sampler = Sampler1D(ranges=y_ranges, num_samples=y_num_samples)

    def grid_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        sample_grid_x = self.x_sampler.grid_sample()
        sample_grid_y = self.y_sampler.grid_sample()
        sample_grid_x, sample_grid_y = torch.meshgrid(
            sample_grid_x, sample_grid_y, indexing="ij"
        )
        return sample_grid_x, sample_grid_y

    def random_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        sample_random_x = self.x_sampler.random_sample(
            num_samples=self.x_sampler.num_samples * self.y_sampler.num_samples
        )
        sample_random_y = self.y_sampler.random_sample(
            num_samples=self.x_sampler.num_samples * self.y_sampler.num_samples
        )

        return sample_random_x, sample_random_y
