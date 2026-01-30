"""Small helper functions used in multiple files."""

from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create a folder if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
