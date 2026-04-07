"""Retry helper with exponential backoff for transient failures."""

from __future__ import annotations

import logging
import time
from urllib.error import URLError

from colony_sdk.client import ColonyAPIError

log = logging.getLogger("colony-agent")

# HTTP status codes that are safe to retry
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def retry_api_call(
    fn,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs,
):
    """Call *fn* with retries on transient Colony API and network errors.

    Retries on:
    - ColonyAPIError with status 429/5xx
    - URLError (connection refused, DNS failure, network unreachable)
    - TimeoutError / OSError

    Non-retryable errors (4xx client errors) are raised immediately.
    Returns None after all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except ColonyAPIError as e:
            last_exc = e
            if e.status not in RETRYABLE_STATUSES:
                raise
            log.warning("API error %s (attempt %d/%d): %s", e.status, attempt + 1, max_retries + 1, e)
        except (URLError, TimeoutError, OSError) as e:
            last_exc = e
            log.warning("Network error (attempt %d/%d): %s", attempt + 1, max_retries + 1, e)

        if attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            log.debug("Retrying in %.1fs...", delay)
            time.sleep(delay)

    log.error("All %d retries exhausted: %s", max_retries + 1, last_exc)
    return None
