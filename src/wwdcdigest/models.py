"""Models for WWDC Digest."""

from typing import Literal

from pydantic import BaseModel


class ImageOptions(BaseModel):
    """Model representing image extraction options."""

    format: Literal["jpg", "png", "avif", "webp"] = "jpg"
    width: int | None = None


class OpenAIConfig(BaseModel):
    """Model representing OpenAI API configuration."""

    api_key: str
    endpoint: str | None = None


class OpenAIResponse(BaseModel):
    """Model representing OpenAI's response format for session summaries."""

    summary: str
    key_points: list[str]


class WWDCFrameSegment(BaseModel):
    """Model representing a frame segment with text and image path."""

    timestamp: str
    text: str
    image_path: str


class WWDCDigest(BaseModel):
    """Model representing a WWDC session digest."""

    session_id: str
    title: str
    summary: str
    key_points: list[str]
    segments: list[WWDCFrameSegment] = []
    markdown_path: str = ""
    language: str = "en"
    source_url: str = ""

    def __str__(self) -> str:
        """Return a string representation of the digest."""
        return f"{self.title} ({self.session_id})"
