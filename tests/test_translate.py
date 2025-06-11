"""Tests for translation utilities."""

from unittest.mock import AsyncMock, patch

import pytest

from wwdcdigest._translate import translate_digest_content
from wwdcdigest.models import OpenAIConfig, WWDCFrameSegment


@pytest.mark.anyio
async def test_translate_digest_content():
    """Test translating digest content to a target language."""
    # Prepare test data
    summary = "This is a test summary"
    key_points = ["Point 1", "Point 2", "Point 3"]
    segments = [
        WWDCFrameSegment(
            timestamp="00:01:23.456",
            text="This is the first segment",
            image_path="/path/to/image1.jpg",
        ),
        WWDCFrameSegment(
            timestamp="00:02:34.567",
            text="This is the second segment",
            image_path="/path/to/image2.jpg",
        ),
    ]
    language = "ja"
    config = OpenAIConfig(api_key="test-key")

    # Mock translate_text to return translated versions
    with patch(
        "wwdcdigest._translate.translate_text", new_callable=AsyncMock
    ) as mock_translate:
        # Set up return values for different inputs
        async def translate_side_effect(text, lang, cfg):  # noqa: ARG001
            translations = {
                "This is a test summary": "これはテストの要約です",
                "Point 1": "ポイント 1",
                "Point 2": "ポイント 2",
                "Point 3": "ポイント 3",
                "This is the first segment": "これは最初のセグメントです",
                "This is the second segment": "これは2番目のセグメントです",
            }
            return translations.get(text, f"Translated: {text}")

        mock_translate.side_effect = translate_side_effect

        # Call the function
        (
            translated_summary,
            translated_key_points,
            translated_segments,
        ) = await translate_digest_content(
            summary, key_points, segments, language, config
        )

        # Check that translation was called for each item
        assert mock_translate.call_count == 1 + len(key_points) + len(segments)

        # Check the results
        assert translated_summary == "これはテストの要約です"
        assert translated_key_points == ["ポイント 1", "ポイント 2", "ポイント 3"]
        assert translated_segments[0].text == "これは最初のセグメントです"
        assert translated_segments[1].text == "これは2番目のセグメントです"

        # Ensure segments were modified in place
        assert segments is translated_segments
        assert segments[0].text == "これは最初のセグメントです"
        assert segments[1].text == "これは2番目のセグメントです"
