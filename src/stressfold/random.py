"""Stable named random seeds used throughout an audit."""

from __future__ import annotations

import hashlib


def derive_seed(root: int, *path: object) -> int:
    """Derive a stable 32-bit seed from a root seed and semantic path."""

    payload = "\x1f".join([str(root), *(str(part) for part in path)]).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8, person=b"stressfd").digest()
    return int.from_bytes(digest[:4], "little", signed=False)
