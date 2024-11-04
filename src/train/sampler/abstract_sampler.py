"""
Abstract class for sampler
"""

from abc import ABCMeta, abstractmethod


class AbstractSampler(metaclass=ABCMeta):
    """
    Abstract class for sampler
    """

    @abstractmethod
    def grid_sample(self, *args, **kwargs):
        """
        Sample data from grid

        Returns:
            tuple[torch.Tensor, torch.Tensor]: x, y
        """
        raise NotImplementedError

    @abstractmethod
    def random_sample(self, *args, **kwargs):
        """
        Sample data from random
        """
        raise NotImplementedError
