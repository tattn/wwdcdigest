"""Translation utilities for WWDC digest."""

import logging

from .models import OpenAIConfig, WWDCFrameSegment
from .openai_utils import translate_text

logger = logging.getLogger("wwdcdigest")


async def translate_digest_content(
    summary: str,
    key_points: list[str],
    segments: list[WWDCFrameSegment],
    language: str,
    config: OpenAIConfig,
) -> tuple[str, list[str], list[WWDCFrameSegment]]:
    """Translate digest content to the target language.

    Args:
        summary: Summary text to translate
        key_points: List of key points to translate
        segments: List of segments with text to translate
        language: Target language code
        config: OpenAI API configuration

    Returns:
        Tuple containing:
        - translated summary
        - translated key points
        - segments with translated text
    """
    logger.info(f"Translating content to {language}")

    # Translate summary
    translated_summary = await translate_text(summary, language, config)

    # Translate key points
    translated_key_points = []
    for point in key_points:
        translated_point = await translate_text(point, language, config)
        translated_key_points.append(translated_point)

    # Translate segment texts (modifies segments in place)
    for segment in segments:
        segment.text = await translate_text(segment.text, language, config)

    return translated_summary, translated_key_points, segments
