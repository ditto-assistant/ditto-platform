"""S3-compatible object store client used by the upload pipeline."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from ditto.api_server.storage.errors import ObjectUploadFailedError
from ditto.api_server.storage.models import StoredObject

if TYPE_CHECKING:
    from types import TracebackType

    from ditto.api_server.storage.models import StorageConfig

logger = logging.getLogger(__name__)


class S3StorageClient:
    """Async wrapper around aioboto3's S3 client.

    Speaks the generic S3 API so the same client works against minio in
    dev compose, AWS S3 in prod, Cloudflare R2, Backblaze B2, or any
    other S3-compatible endpoint via :class:`StorageConfig.endpoint_url`.

    The lifespan owns one of these per process and reuses the underlying
    aioboto3 :class:`aioboto3.Session` across requests. Each call to
    :meth:`put_object` / :meth:`object_exists` opens a short-lived
    ``async with self._session.client("s3", ...)`` block; aiobotocore
    sets up + tears down a fresh client (and TCP connection) per call.
    At MVP scale this is fine; if upload volume ever makes the per-call
    handshake material, cache a long-lived client in :meth:`__aenter__`
    + close in :meth:`__aexit__`.

    Usage:
        async with create_storage_client(config) as storage:
            stored = await storage.put_object(
                key=f"{agent_id}/agent.tar.gz",
                body=tar_bytes,
                content_type="application/gzip",
            )
    """

    def __init__(self, config: StorageConfig) -> None:
        # Lazy import: aioboto3 + boto3 + botocore are heavy. Defer to
        # actual instantiation so import-time cost is paid only by the
        # api_server lifespan, not test discovery.
        import aioboto3

        self._config = config
        self._session = aioboto3.Session(
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name=config.region,
        )

    async def __aenter__(self) -> S3StorageClient:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        return None

    @property
    def public_bucket(self) -> str | None:
        """The transparency-mirror bucket, or ``None`` when publishing is off."""
        return self._config.public_bucket

    async def put_object(
        self,
        *,
        key: str,
        body: bytes,
        content_type: str = "application/octet-stream",
        bucket: str | None = None,
    ) -> StoredObject:
        """Upload ``body`` to ``key``.

        Server-side encryption is enforced at the BUCKET level via
        default-encryption policy rather than per-request, because
        per-request ``ServerSideEncryption`` headers are rejected by
        minio without KMS config. Bucket-level default encryption (set
        via Terraform / mc encrypt for minio / S3 default encryption)
        applies transparently to every object written here.

        Raises:
            ObjectUploadFailedError: When the underlying S3 call raises
                ``botocore.exceptions.ClientError`` or the endpoint is
                unreachable.
        """
        # Lazy: only botocore exceptions are needed here.
        from botocore.exceptions import BotoCoreError, ClientError

        target = bucket or self._config.bucket
        try:
            async with self._session.client(
                "s3",
                endpoint_url=self._config.endpoint_url,
                use_ssl=self._config.use_tls,
            ) as s3:
                await s3.put_object(
                    Bucket=target,
                    Key=key,
                    Body=body,
                    ContentType=content_type,
                )
        except (ClientError, BotoCoreError) as e:
            raise ObjectUploadFailedError(
                f"put_object failed: bucket={target!r} key={key!r} cause={e}"
            ) from e

        sha256 = hashlib.sha256(body).hexdigest()
        logger.info(
            f"stored object bucket={target} key={key} "
            f"size_bytes={len(body)} sha256={sha256}"
        )
        return StoredObject(key=key, size_bytes=len(body), sha256=sha256)

    async def presigned_get_url(self, *, key: str, expires_in: int = 300) -> str:
        """Return a pre-signed GET URL the validator daemon can stream from.

        The URL embeds a time-limited signature, so the bucket can stay
        private while the daemon pulls the tarball directly from object
        storage (no proxying bytes through the API). ``expires_in`` bounds
        the validity window in seconds.

        Generating a pre-signed URL is a local signing operation — no
        network round trip and no existence check — so a URL for a missing
        key is still returned; the daemon's GET then 404s. Callers that need
        existence semantics check the ``agents`` row first.

        Raises:
            ObjectUploadFailedError: When the underlying client cannot be
                constructed or the signing call fails.
        """
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            async with self._session.client(
                "s3",
                endpoint_url=self._config.endpoint_url,
                use_ssl=self._config.use_tls,
            ) as s3:
                return await s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self._config.bucket, "Key": key},
                    ExpiresIn=expires_in,
                )
        except (ClientError, BotoCoreError) as e:
            raise ObjectUploadFailedError(
                f"presigned_get_url failed: bucket={self._config.bucket!r} "
                f"key={key!r} cause={e}"
            ) from e

    async def object_exists(self, *, key: str) -> bool:
        """Return ``True`` iff a HEAD against ``key`` succeeds.

        Used by integration tests + future janitor sweeps. Returns
        ``False`` on 404, raises :class:`ObjectUploadFailedError`-style
        errors for any other failure (wrapped via :class:`StorageError`).

        Raises:
            ObjectUploadFailedError: When the endpoint returns an error
                other than 404.
        """
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            async with self._session.client(
                "s3",
                endpoint_url=self._config.endpoint_url,
                use_ssl=self._config.use_tls,
            ) as s3:
                await s3.head_object(Bucket=self._config.bucket, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise ObjectUploadFailedError(
                f"head_object failed: bucket={self._config.bucket!r} "
                f"key={key!r} cause={e}"
            ) from e
        except BotoCoreError as e:
            raise ObjectUploadFailedError(
                f"head_object failed: bucket={self._config.bucket!r} "
                f"key={key!r} cause={e}"
            ) from e
        return True
