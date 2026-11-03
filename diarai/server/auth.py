"""API-key auth. Only scrypt hashes of keys are stored (DIARAI_API_KEY_HASH, comma-separated for several).

A request presents its key in any of:
    Api-Subscription-Key: <key>                 (Sarvam-style)
    Authorization: Bearer <key>
    Authorization: Basic base64(<any user>:<key>)   (HTTP Basic auth; the username is ignored)
    WebSocket subprotocols ["diarai", "key.<key>"]   (browsers; WebSocket only)

Generate a key:  python -m diarai keygen   (writes the hash to .env, the key to secrets/api_key.txt, mode 600)
"""

import base64
import hashlib
import hmac
import secrets
import threading
import time
from collections import OrderedDict, defaultdict, deque
from typing import Iterable, Mapping, Optional

SCRYPT_N, SCRYPT_R, SCRYPT_P, DKLEN = 2 ** 15, 8, 1, 32
_MAXMEM = 128 * 1024 * 1024


def generate_key() -> str:
    """384 random bits, URL-safe (64 characters)."""
    return secrets.token_urlsafe(48)


def _scrypt(key: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    return hashlib.scrypt(key.encode(), salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=_MAXMEM)


def hash_key(key: str) -> str:
    salt = secrets.token_bytes(16)
    dk = _scrypt(key, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P, DKLEN)
    b64 = lambda b: base64.urlsafe_b64encode(b).decode().rstrip("=")
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${b64(salt)}${b64(dk)}"


def verify_key(key: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, dk = stored.strip().split("$")
        if scheme != "scrypt":
            return False
        unb64 = lambda s: base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
        expected = unb64(dk)
        return hmac.compare_digest(_scrypt(key, unb64(salt), int(n), int(r), int(p), len(expected)), expected)
    except (ValueError, TypeError):
        return False


def extract_key(headers: Mapping[str, str]) -> Optional[str]:
    """The presented key, from Api-Subscription-Key or Authorization (Bearer / Basic). Header names are
    case-insensitive (Starlette's Headers handles that)."""
    key = headers.get("api-subscription-key")
    if key:
        return key.strip()
    auth = headers.get("authorization", "")
    scheme, _, value = auth.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    if scheme.lower() == "basic" and value.strip():
        try:
            _, _, password = base64.b64decode(value.strip(), validate=True).decode().partition(":")
            return password or None
        except (ValueError, UnicodeDecodeError):
            return None
    return None


def extract_ws_key(headers: Mapping[str, str]):
    """(key, subprotocol to accept) for a WebSocket handshake. Browsers cannot set headers on a WebSocket,
    so the page offers subprotocols ["diarai", "key.<KEY>"]; the server accepts "diarai". Other clients
    use the normal headers."""
    offered = [p.strip() for p in headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
    key = next((p[4:] for p in offered if p.startswith("key.")), None) or extract_key(headers)
    return key, ("diarai" if "diarai" in offered else None)


class Authenticator:
    """Checks keys against the stored hashes. scrypt is deliberately slow (~0.1 s), so successful keys are
    remembered (by SHA-256 digest, in memory only) and clients with many failures are throttled."""

    def __init__(self, hashes: Iterable[str], max_failures: int = 20, window_s: float = 300.0,
                 cache_ttl_s: float = 3600.0):
        self.hashes = [h for h in hashes if h.strip()]
        self.max_failures, self.window_s, self.cache_ttl_s = max_failures, window_s, cache_ttl_s
        self._ok: "OrderedDict[bytes, float]" = OrderedDict()
        self._fails = defaultdict(deque)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.hashes)

    def throttled(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._fails[client]
            while q and now - q[0] > self.window_s:
                q.popleft()
            return len(q) >= self.max_failures

    def check(self, key: Optional[str], client: str = "?") -> bool:
        if not key or len(key) > 512:
            self._fail(client)
            return False
        digest = hashlib.sha256(key.encode()).digest()
        now = time.monotonic()
        with self._lock:
            t = self._ok.get(digest)
            if t is not None and now - t < self.cache_ttl_s:
                return True
        if any(verify_key(key, h) for h in self.hashes):
            with self._lock:
                self._ok[digest] = now
                while len(self._ok) > 64:
                    self._ok.popitem(last=False)
            return True
        self._fail(client)
        return False

    def _fail(self, client: str):
        with self._lock:
            self._fails[client].append(time.monotonic())
