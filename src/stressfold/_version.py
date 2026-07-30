"""The single source of the package version.

``reporting`` and ``results`` stamp the version into their artifacts, so the
value lives here rather than being repeated at each use site where copies can
drift apart.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.3.0"
