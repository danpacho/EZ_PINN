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

        # Get samples from each included side
        boundary_sample_x_list = []
        boundary_sample_y_list = []

        top, bottom, left, right = self._sample_side()

        if "t" in side_include:
            boundary_sample_x_list.append(top[0])
            boundary_sample_y_list.append(top[1])
        if "b" in side_include:
            boundary_sample_x_list.append(bottom[0])
            boundary_sample_y_list.append(bottom[1])
        if "l" in side_include:
            boundary_sample_x_list.append(left[0])
            boundary_sample_y_list.append(left[1])
        if "r" in side_include:
            boundary_sample_x_list.append(right[0])
            boundary_sample_y_list.append(right[1])

        # Concatenate all included boundary samples
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

        return (
            domain_sample_x,
            domain_sample_y,
        ), (
            boundary_sample_x,
            boundary_sample_y,
        )

    def _sample_side(self):
        # Define ranges and values for the four sides
        top_range = (self.x_sampler.ranges[0], self.x_sampler.ranges[1])
        top_value = self.y_sampler.ranges[1]

        bottom_range = (self.x_sampler.ranges[0], self.x_sampler.ranges[1])
        bottom_value = self.y_sampler.ranges[0]

        left_range = (self.y_sampler.ranges[0], self.y_sampler.ranges[1])
        left_value = self.x_sampler.ranges[0]

        right_range = (self.y_sampler.ranges[0], self.y_sampler.ranges[1])
        right_value = self.x_sampler.ranges[1]

        # Create samplers for each side
        top_sampler = Sampler2D(
            x_ranges=top_range,
            y_ranges=(top_value, top_value),
            x_num_samples=self.boundary_sample_count_x,
            y_num_samples=1,
        )
        bottom_sampler = Sampler2D(
            x_ranges=bottom_range,
            y_ranges=(bottom_value, bottom_value),
            x_num_samples=self.boundary_sample_count_x,
            y_num_samples=1,
        )
        left_sampler = Sampler2D(
            x_ranges=(left_value, left_value),
            y_ranges=left_range,
            x_num_samples=1,
            y_num_samples=self.boundary_sample_count_y,
        )
        right_sampler = Sampler2D(
            x_ranges=(right_value, right_value),
            y_ranges=right_range,
            x_num_samples=1,
            y_num_samples=self.boundary_sample_count_y,
        )

        # Return samples for each side as tuples of (x, y) coordinates
        return (
            top_sampler.random_sample(),
            bottom_sampler.random_sample(),
            left_sampler.random_sample(),
            right_sampler.random_sample(),
        )
