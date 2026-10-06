"""
Text cleaning and sanitization utility.
"""

import re
import unicodedata


class TextCleaner:
    """Cleans and normalizes raw text for retrieval and chunking."""

    def clean(self, text: str) -> str:
        """Sanitize whitespace, unicode artifacts, and non-printable control characters."""
        if not text:
            return ""

        # Normalize unicode characters (e.g. smart quotes, em-dashes, rupee symbol)
        text = unicodedata.normalize("NFKC", text)

        # Remove zero-width characters and control characters except \n and \t
        text = "".join(ch for ch in text if ch in "\n\t" or not unicodedata.category(ch).startswith("C"))

        # Normalize multiple spaces and tabs to single spaces while keeping paragraph breaks
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        # Collapse 3+ consecutive newlines to 2
        cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))

        return cleaned.strip()
