"""Stateless impression tokens (RFC §4, M1: issue; M2: ingest).

Every served row/page carries a compact HMAC-signed record of WHAT was
shown to WHOM — the anchor for outcome feedback (click → watch → drop)
without any server-side session store. Verification is pure; the token
carries no PII beyond the opaque user id.

Token = base64url(compact JSON payload) + "." + hex(HMAC-SHA256)[:32].
The secret is CINE_REC_IMPRESSION_SECRET; when unset, a random
per-process secret is generated — tokens verify within the process
lifetime, which covers the MVP (feedback ingestion in M2 should set
the env for cross-restart verification).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional, Union

DEFAULT_TTL_SECONDS = 3600
_process_secret: Optional[bytes] = None


def _get_secret() -> bytes:
    global _process_secret
    env = os.getenv("CINE_REC_IMPRESSION_SECRET")
    if env:
        return env.encode()
    if _process_secret is None:
        _process_secret = secrets.token_bytes(32)
    return _process_secret


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64d(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def issue(
    payload: dict,
    ttl: int = DEFAULT_TTL_SECONDS,
    now: Optional[float] = None,
) -> str:
    """Sign a payload dict into a token. exp is embedded; everything
    else in the payload is caller-defined (keep it compact + PII-free)."""
    body = {**payload, "exp": int((now if now is not None else time.time()) + ttl)}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    b64 = _b64e(raw)
    sig = hmac.new(_get_secret(), b64.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{b64}.{sig}"


def verify(
    token: Union[str, None],
    now: Optional[float] = None,
) -> Optional[dict]:
    """The payload dict, or None for malformed / bad-signature / expired
    tokens. Never raises — an unverifiable impression is a no-op."""
    if not token or "." not in token:
        return None
    b64, _, sig = token.rpartition(".")
    expected = hmac.new(_get_secret(), b64.encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(_b64d(b64))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    exp = payload.get("exp")
    if not isinstance(exp, int):
        return None
    if (now if now is not None else time.time()) > exp:
        return None
    return payload
