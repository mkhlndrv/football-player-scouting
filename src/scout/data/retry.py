import time
from collections.abc import Callable

NOT_TRANSIENT = (KeyError, TypeError, ValueError, AttributeError, IndexError)


def until_done[T](
    fn: Callable[[], T],
    *,
    attempts: int = 20,
    wait_s: float = 60,
    log: Callable[[str], None] = print,
) -> T:
    """Providers drop connections at random (Understat every ~120 pages); keep calling.
    Programming errors are not transient and are raised at once."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except NOT_TRANSIENT:
            raise
        except Exception as exc:  # noqa: BLE001 — any transport error is retried
            log(f"attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}")
            if attempt == attempts:
                raise
            time.sleep(wait_s)
    raise RuntimeError("unreachable")
