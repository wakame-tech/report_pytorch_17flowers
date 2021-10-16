import logging
import sys
from typing import Union


def prepare_logger(level):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    logger.addHandler(handler)
    return logger


__logger: Union[logging.Logger, None] = None


def get_logger() -> logging.Logger:
    assert (__logger is not None)
    return __logger


def set_logger(logger: logging.Logger):
    global __logger
    __logger = logger
