"""
models.retry — Retry logic for transient API failures and rate limits.
"""

import os
import time
import random
from typing import Callable

# Bounded retry for transient API failures (5xx, connection resets, timeouts).
_API_MAX_RETRIES = 4
_API_BASE_DELAY = 2.0  # seconds; doubles each retry + jitter

# Long-sleep schedule for HTTP 429 (rate-limit / quota) responses.
_RATE_LIMIT_WAIT_SECONDS = (120, 300, 600)


class RateLimitError(RuntimeError):
    """Raised when a provider responds with HTTP 429 (rate-limit / quota).

    Separate from generic RuntimeError so the episode loop can apply a
    long sleep-and-retry policy rather than recording the episode as crashed.
    """
    pass


def _is_transient_api_error(exc: Exception) -> bool:
    """Decide whether an API error is worth retrying."""
    msg = str(exc).lower()
    transient_markers = (
        "timeout", "timed out", "connection reset", "connection aborted",
        "connection termination", "remote disconnected", "upstream connect",
        "upstream disconnect", "bad gateway", "service unavailable",
        "gateway timeout", "temporarily unavailable",
    )
    if any(m in msg for m in transient_markers):
        return True
    for code in (" 500", " 502", " 503", " 504", " 429"):
        if code in msg:
            return True
    return False


def _check_pause_file() -> None:
    """Block until the shared pause file (if any) expires.

    Used by demo_parallel: when ANY worker hits a rate limit, it writes
    a pause file with a resume timestamp. All workers check this before
    each API call and sleep if the file exists and hasn't expired yet.
    """
    pause_path = os.environ.get("AGENTIC_MQA_PAUSE_FILE", "")
    if not pause_path:
        return
    while os.path.exists(pause_path):
        try:
            with open(pause_path, "r") as f:
                resume_at = float(f.read().strip())
            remaining = resume_at - time.time()
            if remaining <= 0:
                try:
                    os.remove(pause_path)
                except FileNotFoundError:
                    pass
                return
            mins = remaining / 60
            print(f"  [pause] Rate-limit pause active — "
                  f"sleeping {mins:.1f} min until resume...")
            time.sleep(min(remaining, 30.0))
        except (ValueError, OSError):
            try:
                os.remove(pause_path)
            except FileNotFoundError:
                pass
            return


def _write_pause_file(duration_seconds: float) -> None:
    """Write a pause file that tells all workers to sleep."""
    pause_path = os.environ.get("AGENTIC_MQA_PAUSE_FILE", "")
    if not pause_path:
        return
    resume_at = time.time() + duration_seconds
    try:
        with open(pause_path, "w") as f:
            f.write(str(resume_at))
        print(f"  [pause] Wrote pause file ({duration_seconds/60:.0f} min): "
              f"{pause_path}")
    except OSError:
        pass


def retry_api_call(fn: Callable[[], str], label: str) -> str:
    """Call fn() with bounded exponential backoff on transient errors.

    Two separate retry policies:
      1. Generic transient errors (5xx, connection drops, timeouts):
         Quick attempts with exponential backoff. Total wait ~60s.
      2. HTTP 429 (RateLimitError): long sleeps of 2/5/10 min.
         After three failed cycles, re-raises the error.
    """
    rate_limit_attempt = 0
    quick_attempt = 0
    while True:
        _check_pause_file()
        try:
            return fn()
        except RateLimitError as e:
            if rate_limit_attempt >= len(_RATE_LIMIT_WAIT_SECONDS):
                print(f"  [{label}] RATE LIMIT persisted after "
                      f"{rate_limit_attempt} long waits — giving up.")
                raise
            wait = _RATE_LIMIT_WAIT_SECONDS[rate_limit_attempt]
            _write_pause_file(wait)
            mins = wait // 60
            print(f"  [{label}] RATE LIMIT (HTTP 429) — "
                  f"sleeping {mins} min (attempt {rate_limit_attempt + 1}/"
                  f"{len(_RATE_LIMIT_WAIT_SECONDS)}): {str(e)[:200]}")
            time.sleep(wait)
            rate_limit_attempt += 1
        except Exception as e:
            msg = str(e).lower()
            # Detect 429/rate-limit in error text from non-NVIDIA backends
            if (" 429" in msg or "rate limit" in msg
                    or "rate-limit" in msg or "quota" in msg):
                if rate_limit_attempt >= len(_RATE_LIMIT_WAIT_SECONDS):
                    raise
                wait = _RATE_LIMIT_WAIT_SECONDS[rate_limit_attempt]
                mins = wait // 60
                print(f"  [{label}] RATE LIMIT in error text — "
                      f"sleeping {mins} min (attempt {rate_limit_attempt + 1}/"
                      f"{len(_RATE_LIMIT_WAIT_SECONDS)}): {str(e)[:200]}")
                time.sleep(wait)
                rate_limit_attempt += 1
                continue
            if (quick_attempt >= _API_MAX_RETRIES
                    or not _is_transient_api_error(e)):
                raise
            wait = _API_BASE_DELAY * (2 ** quick_attempt) + random.uniform(0, 1.0)
            print(f"  [{label}] transient error (attempt {quick_attempt + 1}/"
                  f"{_API_MAX_RETRIES + 1}), retrying in {wait:.1f}s: "
                  f"{str(e)[:200]}")
            time.sleep(wait)
            quick_attempt += 1
