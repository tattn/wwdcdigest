"""WWDCDigest - Tools for creating digests from Apple WWDC sessions."""

# Re-export public API
from .digest import create_digest
from .models import WWDCDigest

# Version information
__version__ = "0.1.0"

__all__ = [
    "WWDCDigest",
    "create_digest",
]
