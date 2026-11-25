# Base64url encoding for the X-Agent-Hint header (protocol section 3.2).
from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any


class HintDecodeError(ValueError):
    """Raised when a header value cannot be decoded into a JSON object."""


def encode(hint: Mapping[str, Any]) -> str:
    """Serialize a hint into the header value: UTF-8 JSON -> base64url, no padding."""
    raw = json.dumps(
        hint,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode(value: str) -> dict[str, Any]:
    """Decode a header value. Accepts both padded and unpadded base64url."""
    if not isinstance(value, str):
        raise HintDecodeError("header value must be a string")

    text = value.strip()
    if not text:
        raise HintDecodeError("header value is empty")

    padding = "=" * (-len(text) % 4)
    try:
        raw = base64.urlsafe_b64decode(text + padding)
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same
        raise HintDecodeError(f"invalid base64url: {exc}") from exc

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HintDecodeError(f"invalid JSON payload: {exc}") from exc

    if not isinstance(parsed, dict):
        raise HintDecodeError("hint payload must be a JSON object")
    return parsed
