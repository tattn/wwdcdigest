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
from .models import AIConfig, OpenAIConfig
from .summarizer import (
    DefaultSummarizer,
    ExternalAIContentSummarizer,
    OpenAIContentSummarizer,
)
from .translator import ExternalAIContentTranslator, OpenAIContentTranslator
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
        ai_config: AIConfig | OpenAIConfig | None = None,
    ) -> ContentSummarizer:
        """Create a content summarizer implementation.

        Args:
            ai_config: AI configuration (if None, uses DefaultSummarizer)

        Returns:
            Implementation of ContentSummarizer
        """
        if isinstance(ai_config, OpenAIConfig):
            return OpenAIContentSummarizer(ai_config)
        if isinstance(ai_config, AIConfig):
            if ai_config.provider == "openai" and ai_config.api_key:
                return OpenAIContentSummarizer(
                    OpenAIConfig(api_key=ai_config.api_key, endpoint=ai_config.endpoint)
                )
            if ai_config.provider in {"codex", "claude", "command"}:
                return ExternalAIContentSummarizer(ai_config)
        return DefaultSummarizer()

    @staticmethod
    def create_translator(
        ai_config: AIConfig | OpenAIConfig,
    ) -> ContentTranslator:
        """Create a content translator implementation.

        Args:
            ai_config: AI configuration

        Returns:
            Implementation of ContentTranslator
        """
        if isinstance(ai_config, OpenAIConfig):
            return OpenAIContentTranslator(ai_config)
        if ai_config.provider == "openai" and ai_config.api_key:
            return OpenAIContentTranslator(
                OpenAIConfig(api_key=ai_config.api_key, endpoint=ai_config.endpoint)
            )
        if ai_config.provider in {"codex", "claude", "command"}:
            return ExternalAIContentTranslator(ai_config)
        raise ValueError(f"Unsupported translator provider: {ai_config.provider}")

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
