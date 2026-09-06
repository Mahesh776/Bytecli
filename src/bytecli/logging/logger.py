import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from bytecli.logging.formatters import JsonFormatter


def setup_logging(
    level: str | int = "INFO",
    log_file: Path | None = None,
    json_format: bool = False,
    verbose: bool = False,
    rotation: str = "10 MB",
    retention: int = 5,
    console: bool = True,
) -> None:
    logger.remove()

    log_level: str
    if verbose:
        log_level = "DEBUG"
    else:
        log_level = level if isinstance(level, str) else logging.getLevelName(level) or "INFO"

    if console:
        if json_format:
            fmt: Callable[[dict[str, Any]], str] = JsonFormatter()
            logger.add(
                sys.stderr,
                level=log_level,
                format=fmt,  # type: ignore[arg-type]
                colorize=False,
            )
        else:
            logger.add(
                sys.stderr,
                level=log_level,
                format=(
                    "<level>{level: <8}</level> | <cyan>{name}</cyan>:"
                    "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
                ),
                colorize=True,
                diagnose=verbose,
            )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if json_format:
            fmt = JsonFormatter()
            logger.add(
                str(log_file),
                level=log_level,
                format=fmt,  # type: ignore[arg-type]
                rotation=rotation,
                retention=retention,
                colorize=False,
            )
        else:
            logger.add(
                str(log_file),
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation=rotation,
                retention=retention,
                colorize=False,
            )

    _patch_stdlib_logging()


def _patch_stdlib_logging() -> None:
    class LoguruHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                log_level: str | int = logger.level(record.levelname).name
            except ValueError:
                log_level = record.levelno
            frame = logging.currentframe()
            depth = 2
            while frame is not None and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back  # type: ignore[assignment]
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(log_level, record.getMessage())

    logging.basicConfig(handlers=[LoguruHandler()], level=logging.DEBUG, force=True)


def get_logger(name: str | None = None) -> Any:
    if name:
        return logger.bind(module=name)
    return logger
