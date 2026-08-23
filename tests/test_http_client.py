import pytest

from pivotrag.http_client import _headers, _normalize_base_url


def test_headers_omit_authorization_without_api_key() -> None:
    assert _headers("") == {"Content-Type": "application/json"}


def test_headers_add_authorization_for_configured_api_key() -> None:
    assert _headers(" example-key ")["Authorization"] == "Bearer example-key"


def test_base_url_is_required() -> None:
    with pytest.raises(ValueError, match="BASE_URL"):
        _normalize_base_url("  ")


def test_base_url_normalization() -> None:
    assert _normalize_base_url("https://example.test/v1/") == "https://example.test"
