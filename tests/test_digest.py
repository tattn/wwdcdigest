"""Tests for the digest module."""

import pytest

from wwdcdigest.digest import create_digest
from wwdcdigest.models import OpenAIConfig, WWDCDigest


@pytest.mark.anyio
async def test_wwdc_digest_model():
    """Test the WWDCDigest model."""
    # Create a test digest
    digest = WWDCDigest(
        session_id="110173",
        title="Test Session",
        summary="This is a test summary",
        key_points=["Point 1", "Point 2", "Point 3"],
        source_url="https://developer.apple.com/videos/play/wwdc2023/110173/",
    )

    # Check that the model is created correctly
    assert digest.session_id == "110173"
    assert digest.title == "Test Session"
    assert digest.summary == "This is a test summary"
    assert len(digest.key_points) == 3
    assert "Point 1" in digest.key_points
    assert "Point 2" in digest.key_points
    assert "Point 3" in digest.key_points

    # Check string representation
    assert str(digest) == "Test Session (110173)"


# Test will be implemented when the actual digest creation logic is implemented
@pytest.mark.anyio
@pytest.mark.skip(reason="Requires internet connection and actual WWDC session data")
async def test_create_digest():
    """Test creating a digest from a session URL."""
    url = "https://developer.apple.com/videos/play/wwdc2023/10149/"
    digest = await create_digest(url)
    assert digest.session_id is not None
    assert digest.title is not None
    assert digest.summary is not None
    assert isinstance(digest.key_points, list)
    # Without OpenAI key, key_points should be empty
    assert len(digest.key_points) == 0


@pytest.mark.anyio
@pytest.mark.skip(
    reason="Requires internet connection, actual WWDC session data, and OpenAI key"
)
async def test_create_digest_with_openai():
    """Test creating a digest from a session URL with OpenAI key."""
    url = "https://developer.apple.com/videos/play/wwdc2023/10149/"
    openai_config = OpenAIConfig(api_key="test_key")
    digest = await create_digest(url, openai_config=openai_config)
    assert digest.session_id is not None
    assert digest.title is not None
    assert digest.summary is not None
    assert len(digest.key_points) > 0
