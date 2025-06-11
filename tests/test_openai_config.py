"""Tests for OpenAI configuration utilities."""

import os
from unittest.mock import patch

import pytest

from wwdcdigest.digest import _validate_openai_settings
from wwdcdigest.models import OpenAIConfig


def test_validate_openai_settings_with_params():
    """Test validating OpenAI settings with explicit parameters."""
    # Test with both key and endpoint provided
    config_input = OpenAIConfig(api_key="test-key", endpoint="https://test-endpoint")
    config = _validate_openai_settings(config_input, "en")
    assert isinstance(config, OpenAIConfig)
    assert config.api_key == "test-key"
    assert config.endpoint == "https://test-endpoint"

    # Test with only key provided
    config_input = OpenAIConfig(api_key="test-key", endpoint=None)
    config = _validate_openai_settings(config_input, "en")
    assert isinstance(config, OpenAIConfig)
    assert config.api_key == "test-key"
    assert config.endpoint is None


@patch.dict(
    os.environ, {"OPENAI_API_KEY": "env-key", "OPENAI_API_ENDPOINT": "env-endpoint"}
)
def test_validate_openai_settings_with_env_vars():
    """Test validating OpenAI settings with environment variables."""
    # Test with environment variables
    config = _validate_openai_settings(None, "en")
    assert isinstance(config, OpenAIConfig)
    assert config.api_key == "env-key"
    assert config.endpoint == "env-endpoint"


def test_validate_openai_settings_non_english_no_key():
    """Test that an error is raised when requesting non-English without an API key."""
    with pytest.raises(
        ValueError, match="OpenAI API key is required for non-English languages"
    ):
        _validate_openai_settings(None, "ja")


def test_validate_openai_settings_english_no_key():
    """Test that None is returned for English with no API key."""
    config = _validate_openai_settings(None, "en")
    assert config is None
