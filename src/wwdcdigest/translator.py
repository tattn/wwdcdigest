"""Implementations of content translation components."""

import logging

from .interfaces import OpenAITranslator
from .models import WWDCFrameSegment
from .openai_utils import translate_text

logger = logging.getLogger("wwdcdigest")


class OpenAIContentTranslator(OpenAITranslator):
    """Implementation of ContentTranslator using OpenAI API."""

    async def translate(
        self,
        summary: str,
        key_points: list[str],
        segments: list[WWDCFrameSegment],
        target_language: str,
    ) -> tuple[str, list[str], list[WWDCFrameSegment]]:
        """Translate digest content to the target language using OpenAI.

        Args:
            summary: The summary text to translate
            key_points: List of key points to translate
            segments: List of WWDCFrameSegment objects with text to translate
            target_language: Target language code

        Returns:
            Tuple of (translated_summary, translated_key_points, translated_segments)
        """
        logger.info(f"Translating content to {target_language} using OpenAI")

        # Translate summary
        translated_summary = await translate_text(summary, target_language, self.config)

        # Translate key points
        translated_key_points = []
        for point in key_points:
            translated_point = await translate_text(point, target_language, self.config)
            translated_key_points.append(translated_point)

        # Translate segment text
        for segment in segments:
            segment.text = await translate_text(
                segment.text, target_language, self.config
            )

        return translated_summary, translated_key_points, segments
