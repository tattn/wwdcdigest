"""Utility functions for the CLI."""

from urllib.parse import urlparse

import click


def validate_session_id_or_url(
    _ctx: click.Context, _param: click.Parameter, value: str | None
) -> str | None:
    """Validate that the input is a valid WWDC session URL.

    Args:
        _ctx: Click context (unused)
        _param: Click parameter (unused)
        value: The URL to validate

    Returns:
        The validated URL

    Raises:
        click.BadParameter: If the input is invalid
    """
    if not value:
        return value

    # Check if it's a URL
    if value.startswith(("http://", "https://")):
        parsed_url = urlparse(value)
        if (
            parsed_url.netloc == "developer.apple.com"
            and "/videos/play/wwdc" in parsed_url.path
        ):
            return value
        message = (
            f"URL must be a valid WWDC session URL from developer.apple.com, "
            f"got {value}"
        )
        raise click.BadParameter(message)

    # Not a valid URL
    message = (
        f"Input must be a valid WWDC session URL from developer.apple.com, got {value}"
    )
    raise click.BadParameter(message)
