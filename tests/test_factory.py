"""Tests for the factory module."""

import pytest

from wwdcdigest.factory import DigestComponentFactory
from wwdcdigest.formatter import MarkdownFormatter
from wwdcdigest.models import OpenAIConfig
from wwdcdigest.summarizer import DefaultSummarizer, OpenAIContentSummarizer
from wwdcdigest.translator import OpenAIContentTranslator
from wwdcdigest.video_processor import DefaultVideoProcessor


def test_create_video_processor():
    """Test creating a video processor."""
    processor = DigestComponentFactory.create_video_processor()
    assert isinstance(processor, DefaultVideoProcessor)


def test_create_summarizer_default():
    """Test creating a default summarizer."""
    summarizer = DigestComponentFactory.create_summarizer()
    assert isinstance(summarizer, DefaultSummarizer)


def test_create_summarizer_openai():
    """Test creating an OpenAI summarizer."""
    config = OpenAIConfig(api_key="test-key")
    summarizer = DigestComponentFactory.create_summarizer(config)
    assert isinstance(summarizer, OpenAIContentSummarizer)


def test_create_translator():
    """Test creating a translator."""
    config = OpenAIConfig(api_key="test-key")
    translator = DigestComponentFactory.create_translator(config)
    assert isinstance(translator, OpenAIContentTranslator)


def test_create_formatter_markdown():
    """Test creating a markdown formatter."""
    formatter = DigestComponentFactory.create_formatter("markdown")
    assert isinstance(formatter, MarkdownFormatter)


def test_create_formatter_unsupported():
    """Test creating an unsupported formatter type."""
    with pytest.raises(ValueError, match="Unsupported format type"):
        # This is a type error, but we're testing the runtime behavior
        DigestComponentFactory.create_formatter("html")  # type: ignore
