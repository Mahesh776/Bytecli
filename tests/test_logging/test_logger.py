import sys
from pathlib import Path

import pytest
from loguru import logger

from bytecli.logging.logger import setup_logging


def _stderr_sinks() -> int:
    count = 0
    for handler in logger._core.handlers.values():  # type: ignore[attr-defined]
        sink = handler._sink  # type: ignore[attr-defined]
        stream = getattr(sink, "_stream", sink)
        if stream is sys.stderr:
            count += 1
    return count


@pytest.fixture(autouse=True)
def _reset_logger() -> None:
    logger.remove()


def test_setup_logging_console_enabled_adds_stderr_sink() -> None:
    setup_logging(console=True)
    assert _stderr_sinks() == 1


def test_setup_logging_console_disabled_removes_stderr_sink() -> None:
    setup_logging(console=True)
    assert _stderr_sinks() == 1
    setup_logging(console=False)
    assert _stderr_sinks() == 0


def test_setup_logging_console_disabled_writes_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "bytecli.log"
    setup_logging(console=False, log_file=log_file)
    assert _stderr_sinks() == 0
    logger.info("TUI mode message")
    assert "TUI mode message" in log_file.read_text()
