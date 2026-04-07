"""Tests for colony_agent.retry."""

from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest
from colony_sdk.client import ColonyAPIError

from colony_agent.retry import retry_api_call


class TestRetryApiCall:
    def test_success_no_retry(self):
        fn = MagicMock(return_value={"ok": True})
        result = retry_api_call(fn, "arg1", key="val")
        assert result == {"ok": True}
        fn.assert_called_once_with("arg1", key="val")

    def test_retries_on_429(self):
        fn = MagicMock(side_effect=[
            ColonyAPIError("rate limited", status=429),
            {"ok": True},
        ])
        with patch("colony_agent.retry.time.sleep"):
            result = retry_api_call(fn, max_retries=3)
        assert result == {"ok": True}
        assert fn.call_count == 2

    def test_retries_on_500(self):
        fn = MagicMock(side_effect=[
            ColonyAPIError("server error", status=500),
            ColonyAPIError("server error", status=502),
            {"recovered": True},
        ])
        with patch("colony_agent.retry.time.sleep"):
            result = retry_api_call(fn, max_retries=3)
        assert result == {"recovered": True}
        assert fn.call_count == 3

    def test_does_not_retry_4xx(self):
        fn = MagicMock(side_effect=ColonyAPIError("not found", status=404))
        with pytest.raises(ColonyAPIError) as exc_info:
            retry_api_call(fn, max_retries=3)
        assert exc_info.value.status == 404
        fn.assert_called_once()

    def test_does_not_retry_403(self):
        fn = MagicMock(side_effect=ColonyAPIError("forbidden", status=403))
        with pytest.raises(ColonyAPIError):
            retry_api_call(fn, max_retries=3)
        fn.assert_called_once()

    def test_retries_on_url_error(self):
        fn = MagicMock(side_effect=[
            URLError("connection refused"),
            {"ok": True},
        ])
        with patch("colony_agent.retry.time.sleep"):
            result = retry_api_call(fn, max_retries=2)
        assert result == {"ok": True}
        assert fn.call_count == 2

    def test_retries_on_timeout(self):
        fn = MagicMock(side_effect=[
            TimeoutError("timed out"),
            {"ok": True},
        ])
        with patch("colony_agent.retry.time.sleep"):
            result = retry_api_call(fn, max_retries=2)
        assert result == {"ok": True}

    def test_retries_on_os_error(self):
        fn = MagicMock(side_effect=[
            OSError("network unreachable"),
            {"ok": True},
        ])
        with patch("colony_agent.retry.time.sleep"):
            result = retry_api_call(fn, max_retries=2)
        assert result == {"ok": True}

    def test_returns_none_after_exhausted(self):
        fn = MagicMock(side_effect=ColonyAPIError("down", status=503))
        with patch("colony_agent.retry.time.sleep"):
            result = retry_api_call(fn, max_retries=2)
        assert result is None
        assert fn.call_count == 3  # initial + 2 retries

    def test_exponential_backoff(self):
        fn = MagicMock(side_effect=[
            ColonyAPIError("error", status=500),
            ColonyAPIError("error", status=500),
            {"ok": True},
        ])
        with patch("colony_agent.retry.time.sleep") as mock_sleep:
            retry_api_call(fn, max_retries=3, base_delay=1.0)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)  # 1.0 * 2^0
        mock_sleep.assert_any_call(2.0)  # 1.0 * 2^1

    def test_no_sleep_on_success(self):
        fn = MagicMock(return_value="ok")
        with patch("colony_agent.retry.time.sleep") as mock_sleep:
            retry_api_call(fn, max_retries=3)
        mock_sleep.assert_not_called()

    def test_max_retries_zero(self):
        fn = MagicMock(side_effect=ColonyAPIError("down", status=503))
        with patch("colony_agent.retry.time.sleep"):
            result = retry_api_call(fn, max_retries=0)
        assert result is None
        fn.assert_called_once()
