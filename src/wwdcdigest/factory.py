"""Factory classes for creating WWDC Digest components."""

import logging
from typing import Literal

from .formatter import MarkdownFormatter
from .interfaces import (
    ContentSummarizer,
    ContentTranslator,
    DigestFormatter,
    VideoProcessor,
)
from .models import OpenAIConfig
from .summarizer import DefaultSummarizer, OpenAIContentSummarizer
from .translator import OpenAIContentTranslator
from .video_processor import DefaultVideoProcessor

logger = logging.getLogger("wwdcdigest")


class DigestComponentFactory:
    """Factory for creating WWDC Digest components."""

    @staticmethod
    def create_video_processor() -> VideoProcessor:
        """Create a video processor implementation.

        Returns:
            Implementation of VideoProcessor
        """
        return DefaultVideoProcessor()

    @staticmethod
    def create_summarizer(
        openai_config: OpenAIConfig | None = None,
    ) -> ContentSummarizer:
        """Create a content summarizer implementation.

        Args:
            openai_config: OpenAI API configuration (if None, uses DefaultSummarizer)

        Returns:
            Implementation of ContentSummarizer
        """
        if openai_config:
            return OpenAIContentSummarizer(openai_config)
        return DefaultSummarizer()

    @staticmethod
    def create_translator(
        openai_config: OpenAIConfig,
    ) -> ContentTranslator:
        """Create a content translator implementation.

        Args:
            openai_config: OpenAI API configuration

        Returns:
            Implementation of ContentTranslator
        """
        return OpenAIContentTranslator(openai_config)

    @staticmethod
    def create_formatter(
        format_type: Literal["markdown"] = "markdown",
    ) -> DigestFormatter:
        """Create a digest formatter implementation.

        Args:
            format_type: Type of formatter to create

        Returns:
            Implementation of DigestFormatter
        """
        if format_type == "markdown":
            return MarkdownFormatter()

        # If other formats are added in the future, they would be handled here
        raise ValueError(f"Unsupported format type: {format_type}")
