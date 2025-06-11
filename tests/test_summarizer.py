"""Tests for the summarizer module."""

from unittest.mock import AsyncMock, patch

import pytest

from wwdcdigest.models import OpenAIConfig
from wwdcdigest.summarizer import DefaultSummarizer, OpenAIContentSummarizer


@pytest.mark.anyio
async def test_default_summarizer():
    """Test the DefaultSummarizer."""
    summarizer = DefaultSummarizer()
    transcript = "This is a test transcript"
    session_title = "Test Session"

    summary, key_points = await summarizer.generate_summary(
        transcript, session_title, "en"
    )

    assert summary == "Summary of Test Session"
    assert key_points == []


@pytest.mark.anyio
async def test_openai_summarizer():
    """Test the OpenAIContentSummarizer."""
    config = OpenAIConfig(api_key="test-key")
    summarizer = OpenAIContentSummarizer(config)
    transcript = "This is a test transcript"
    session_title = "Test Session"

    # Mock the OpenAI call
    with patch(
        "wwdcdigest.summarizer.generate_summary_and_key_points",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = (
            "Test summary from OpenAI",
            ["Point 1", "Point 2"],
        )

        summary, key_points = await summarizer.generate_summary(
            transcript, session_title, "en"
        )

        # Check that the mock was called with the right arguments
        mock_generate.assert_called_once_with(transcript, session_title, config, "en")

        # Check the results
        assert summary == "Test summary from OpenAI"
        assert key_points == ["Point 1", "Point 2"]
