from typing import Literal
import torch

from src.train.sampler.sampler_2d import Sampler2D


class RectBoundarySampler(Sampler2D):
    """
    Sample from the boundary of a rectangle

    1. Consists of 4 sides: top, bottom, left, right = 1D Sampler
    2. Sample from domain: 2D Sampler
    """

    def __init__(
        self,
        domain_x: tuple[float, float, int],
        domain_y: tuple[float, float, int],
        boundary_sample_count: tuple[int, int],
    ):
        super().__init__(
            x_ranges=(domain_x[0], domain_x[1]),
            y_ranges=(domain_y[0], domain_y[1]),
            x_num_samples=domain_x[2],
            y_num_samples=domain_y[2],
        )

        self.boundary_sample_count_x = boundary_sample_count[0]
        self.boundary_sample_count_y = boundary_sample_count[1]

    def sample_rect(
        self,
        side_include: list[Literal["t", "l", "b", "r"]] = ["t", "l", "b", "r"],
    ):
        """
        Sample from the domain and boundary of the rectangle

        Args:
            side_include (list[Literal["t", "l", "b", "r"]], optional):
                Boundary sampling inclusion option.
                Defaults to ["t", "l", "b", "r"].

        Returns:
            tuple: `(domain_sample_x, domain_sample_y), (boundary_sample_x, boundary_sample_y)`
        """
        # Get random domain sample
        domain_sample_x, domain_sample_y = self.random_sample()

        # Get combined boundary samples for specified sides
        boundary_sample_x, boundary_sample_y = self._sample_multiple_side(side_include)

        return (domain_sample_x, domain_sample_y), (
            boundary_sample_x,
            boundary_sample_y,
        )

    def _sample_one_side(self, side: Literal["t", "l", "b", "r"]) -> torch.Tensor:
        """
        Sample from the boundary of one side of the rectangle

        Args:
            side (Literal["t", "l", "b", "r"]): Side to sample from

        Returns:
            torch.Tensor: Sampled x and y coordinates
        """
        if side == "t":
            x_ranges = (self.x_sampler.ranges[0], self.x_sampler.ranges[1])
            y_ranges = (self.y_sampler.ranges[1], self.y_sampler.ranges[1])
            x_num_samples = self.boundary_sample_count_x
            y_num_samples = 1
        elif side == "b":
            x_ranges = (self.x_sampler.ranges[0], self.x_sampler.ranges[1])
            y_ranges = (self.y_sampler.ranges[0], self.y_sampler.ranges[0])
            x_num_samples = self.boundary_sample_count_x
            y_num_samples = 1
        elif side == "l":
            x_ranges = (self.x_sampler.ranges[0], self.x_sampler.ranges[0])
            y_ranges = (self.y_sampler.ranges[0], self.y_sampler.ranges[1])
            x_num_samples = 1
            y_num_samples = self.boundary_sample_count_y
        elif side == "r":
            x_ranges = (self.x_sampler.ranges[1], self.x_sampler.ranges[1])
            y_ranges = (self.y_sampler.ranges[0], self.y_sampler.ranges[1])
            x_num_samples = 1
            y_num_samples = self.boundary_sample_count_y
        else:
            raise ValueError(
                f"Invalid side '{side}' specified. Choose from 't', 'b', 'l', or 'r'."
            )

        # Create and return a sample for the specified side
        sampler = Sampler2D(
            x_ranges=x_ranges,
            y_ranges=y_ranges,
            x_num_samples=x_num_samples,
            y_num_samples=y_num_samples,
        )
        return sampler.random_sample()

    def _sample_multiple_side(
        self, sides: list[Literal["t", "l", "b", "r"]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from multiple specified sides of the rectangle and return combined boundary samples.

        Args:
            sides (list[Literal["t", "l", "b", "r"]]): List of sides to sample from.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Combined x and y coordinates for the sampled boundaries.
        """
        boundary_sample_x_list = []
        boundary_sample_y_list = []

        for side in sides:
            side_sample = self._sample_one_side(side)
            boundary_sample_x_list.append(side_sample[0])
            boundary_sample_y_list.append(side_sample[1])

        # Concatenate all samples for the specified sides
        boundary_sample_x = (
            torch.cat(boundary_sample_x_list, dim=0)
            if boundary_sample_x_list
            else torch.tensor([])
        )
        boundary_sample_y = (
            torch.cat(boundary_sample_y_list, dim=0)
            if boundary_sample_y_list
            else torch.tensor([])
        )

        return boundary_sample_x, boundary_sample_y

    def _sample_side(self):
        """
        Sample from all sides of the rectangle

        Returns:
            tuple: (top, bottom, left, right)
        """
        return (
            self._sample_one_side("t"),
            self._sample_one_side("b"),
            self._sample_one_side("l"),
            self._sample_one_side("r"),
        )
