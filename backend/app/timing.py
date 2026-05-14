import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("timing")

_start: dict[str, float] = {}


def t(label: str) -> None:
    """Log elapsed time since last call to t(), and since session start."""
    now = time.perf_counter()
    if "session" not in _start:
        _start["session"] = now
    elapsed_total = round(now - _start["session"], 2)
    elapsed_step = round(now - _start.get("last", _start["session"]), 2)
    _start["last"] = now
    log.info(f"[+{elapsed_step:6.2f}s | total {elapsed_total:6.2f}s]  {label}")


def reset() -> None:
    _start.clear()
