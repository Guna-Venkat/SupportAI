"""
version.py
==========
Package version and build metadata helper.

Provides absolute visibility over the active build: package version,
git commit hash, and package compilation or run-time initialization timestamp.
"""

import subprocess
from datetime import datetime

# Central version identifier
__version__ = "0.1.0"


def get_git_commit_hash() -> str:
    """Retrieves the active Git commit hash.

    Attempts to call `git rev-parse HEAD`. Fallback to environment variable
    or "unknown" if Git is unavailable or outside a repository.

    Returns:
        The SHA-1 hash of the current commit, or "unknown".
    """
    try:
        # Check standard subprocess output
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def get_build_timestamp() -> str:
    """Returns the initialization/build timestamp.

    Returns:
        ISO 8601 formatted timestamp string.
    """
    return datetime.now().isoformat()
