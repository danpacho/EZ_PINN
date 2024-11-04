import logging
from enum import Enum


class LogLevel(Enum):
    """
    Enum class for log levels.
    """

    DEBUG = logging.DEBUG  # 10
    INFO = logging.INFO  # 20
    WARNING = logging.WARNING  # 30
    ERROR = logging.ERROR  # 40
    CRITICAL = logging.CRITICAL  # 50


def create_logger(name: str, level: LogLevel) -> logging.Logger:
    """
    Create a logger with the specified name and level.

    Args:
        name (str): logger_name
        level (LogLevel): logger_level

    Returns:
        logging.Logger: _description_
    """
    # Create a named logger
    logger = logging.getLogger(f"@{name}")
    logger.setLevel(level.value)  # Set logger level using enum value

    # Create a console handler and set its level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level.value)

    # Set the formatter for the console handler
    formatter = logging.Formatter(
        "\n› [%(asctime)s:%(name)s:%(levelname)s]\n› %(message)s",
        datefmt="%I:%M:%S%p",
    )
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger
