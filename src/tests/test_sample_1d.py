from src.tests.logger import logger
from src.train.sampler.sampler_1d import Sampler1D


def test_sample_1d():

    sampler = Sampler1D(ranges=(0, 1), num_samples=10)
    grid_sample = sampler.grid_sample()
    random_sample = sampler.random_sample()
    assert grid_sample.shape == (10,)
    assert random_sample.shape == (10,)

    logger.debug("Sample 1D test passed")

    logger.debug(f"Grid sample: {grid_sample}")
    logger.debug(f"Random sample: {random_sample}")
