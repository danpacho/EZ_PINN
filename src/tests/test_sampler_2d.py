from src.tests.logger import logger
from src.train.sampler.sampler_2d import Sampler2D


def test_sampler2d():
    sampler = Sampler2D(
        x_ranges=(0, 1), x_num_samples=3, y_ranges=(0, 1), y_num_samples=3
    )
    grid_sample = sampler.grid_sample()
    random_sample = sampler.random_sample()
    assert grid_sample.shape == (3, 3, 2)
    assert random_sample.shape == (3, 3, 2)

    logger.debug("Sampler 2D test passed")

    logger.debug(f"Grid sample: {grid_sample}")
    logger.debug(f"Random sample: {random_sample}")

    x, y = grid_sample[..., 0], grid_sample[..., 1]

    logger.debug(f"x: {x}")
    logger.debug(f"y: {y}")
