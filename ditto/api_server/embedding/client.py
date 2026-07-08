"""L3c code-embedding client: a thin, best-effort wrapper over a TEI service.

The embedding is a review-band moderation signal computed once per upload, so the
client never raises into the caller: any failure (service down, timeout, malformed
response) degrades to ``None`` — "no embedding", read downstream as no L3c signal —
exactly like the pure fingerprint extractors. The gate does the cosine comparison
later, in Python, over the small eligible ledger; the model is only hit at embed
time.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Protocol

import httpx

if TYPE_CHECKING:
    from ditto.api_server.embedding.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Surface the upload path and factory rely on."""

    @property
    def model_tag(self) -> str | None:
        """``model@revision`` stored with each vector, or ``None`` when disabled."""
        ...

    async def embed(self, text: str) -> list[float] | None:
        """Return the unit-norm embedding of ``text``, or ``None`` (best-effort)."""
        ...

    async def aclose(self) -> None:
        """Release any underlying connection pool."""
        ...


class NullEmbedder:
    """The disabled embedder: embeds nothing, so the L3c column stays null.

    Used whenever ``L3C_EMBEDDER_URL`` is unset, so the platform runs unchanged
    until an operator points it at a live service.
    """

    model_tag: str | None = None

    async def embed(self, text: str) -> list[float] | None:
        del text  # disabled: nothing is embedded
        return None

    async def aclose(self) -> None:
        return None


class TeiEmbedder:
    """Best-effort client for a text-embeddings-inference ``/embed`` endpoint.

    Requests a unit-norm vector; when ``config.dim`` is set it truncates to that
    Matryoshka width and renormalizes so cosine stays a dot product. Every failure
    path returns ``None`` and logs at INFO — the upload must never fail because the
    moderation embedder is unavailable.
    """

    def __init__(self, config: EmbeddingConfig, client: httpx.AsyncClient) -> None:
        self._config = config
        self._client = client

    @property
    def model_tag(self) -> str | None:
        return self._config.model_tag

    async def embed(self, text: str) -> list[float] | None:
        if not text:
            return None
        url = self._config.url
        assert url is not None  # a TeiEmbedder is only built when enabled
        try:
            resp = await self._client.post(
                f"{url.rstrip('/')}/embed",
                json={"inputs": text, "normalize": True, "truncate": True},
            )
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.info("l3c-embed: request failed (%s)", type(e).__name__)
            return None
        vector = _first_vector(payload)
        if vector is None:
            logger.info("l3c-embed: unexpected response shape")
            return None
        if self._config.dim is not None:
            vector = _l2_normalize(vector[: self._config.dim])
        return vector

    async def aclose(self) -> None:
        await self._client.aclose()


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Return a :class:`TeiEmbedder` when enabled, else the :class:`NullEmbedder`.

    Caller owns lifecycle: ``await embedder.aclose()`` once finished (the api_server
    lifespan registers it on the ``AsyncExitStack``).
    """
    if not config.enabled:
        return NullEmbedder()
    client = httpx.AsyncClient(
        timeout=config.timeout_seconds,
        headers={"User-Agent": "ditto-api-server/0.0.1"},
    )
    return TeiEmbedder(config, client)


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity of two vectors in ``[-1, 1]``; ``0.0`` on any missing input.

    Returns ``0.0`` — read as "no L3c match" — when either vector is ``None``,
    empty, of mismatched length, or has zero norm, so callers can threshold without
    special-casing. Pure + deterministic.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _first_vector(payload: object) -> list[float] | None:
    """Extract the single embedding from a TEI ``/embed`` response.

    TEI returns a list of vectors, one per input; we send one string, so the vector
    is ``payload[0]``. Tolerates a bare ``[float, ...]`` too. Returns ``None`` on any
    other shape or a non-numeric element.
    """
    if not isinstance(payload, list) or not payload:
        return None
    inner = payload[0]
    if isinstance(inner, list):  # [[...]] — the normal case
        candidate = inner
    elif isinstance(inner, (int, float)):  # [...] — a bare vector
        candidate = payload
    else:
        return None
    if not candidate or not all(isinstance(x, (int, float)) for x in candidate):
        return None
    return [float(x) for x in candidate]


def _l2_normalize(vector: list[float]) -> list[float]:
    """Return the unit-norm form of ``vector`` (unchanged if its norm is zero)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]
