"""Tests for the translator module."""

from unittest.mock import AsyncMock, patch

import pytest

from wwdcdigest.models import OpenAIConfig, WWDCFrameSegment
from wwdcdigest.translator import OpenAIContentTranslator


@pytest.mark.anyio
async def test_openai_translator():
    """Test the OpenAIContentTranslator."""
    config = OpenAIConfig(api_key="test-key")
    translator = OpenAIContentTranslator(config)

    # Test data
    summary = "This is a test summary"
    key_points = ["Point 1", "Point 2"]
    segments = [
        WWDCFrameSegment(
            timestamp="00:01:23.456",
            text="This is segment 1",
            image_path="/path/to/image1.jpg",
        ),
        WWDCFrameSegment(
            timestamp="00:02:34.567",
            text="This is segment 2",
            image_path="/path/to/image2.jpg",
        ),
    ]
    target_language = "ja"

    # Mock the translate_text function
    with patch(
        "wwdcdigest.translator.translate_text", new_callable=AsyncMock
    ) as mock_translate:
        # Set up return values for different inputs
        async def translate_side_effect(text, lang, cfg):  # noqa: ARG001
            translations = {
                "This is a test summary": "これはテストの要約です",
                "Point 1": "ポイント 1",
                "Point 2": "ポイント 2",
                "This is segment 1": "これはセグメント1です",
                "This is segment 2": "これはセグメント2です",
            }
            return translations.get(text, f"Translated: {text}")

        mock_translate.side_effect = translate_side_effect

        # Call the translate method
        (
            translated_summary,
            translated_key_points,
            translated_segments,
        ) = await translator.translate(summary, key_points, segments, target_language)

        # Check the number of calls
        assert mock_translate.call_count == 1 + len(key_points) + len(segments)

        # Check the results
        assert translated_summary == "これはテストの要約です"
        assert translated_key_points == ["ポイント 1", "ポイント 2"]
        assert translated_segments[0].text == "これはセグメント1です"
        assert translated_segments[1].text == "これはセグメント2です"

        # Check that segments were modified in place
        assert segments is translated_segments
        assert segments[0].text == "これはセグメント1です"
        assert segments[1].text == "これはセグメント2です"
