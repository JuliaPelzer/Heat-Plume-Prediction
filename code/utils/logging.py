import logging
from typing import Any


def debug(*messages: Any) -> None:
    message = " ".join([str(msg) for msg in messages])
    logging.debug(message)


def info(*messages: Any) -> None:
    message = " ".join([str(msg) for msg in messages])
    logging.info(message)


def warning(*messages: Any) -> None:
    message = " ".join([str(msg) for msg in messages])
    logging.warning(message)


def error(*messages: Any) -> None:
    message = " ".join([str(msg) for msg in messages])
    logging.error(message)
    exit(1)


def set_log_level_debug() -> None:
    logging.getLogger().setLevel(logging.DEBUG)


def set_log_level_info() -> None:
    logging.getLogger().setLevel(logging.INFO)
