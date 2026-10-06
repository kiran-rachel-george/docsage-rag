"""N1: Langfuse tracing. Everything here is a no-op when no keys are configured, so the
app, tests and CI run unchanged without Langfuse."""
import re
from contextlib import contextmanager, nullcontext
from functools import lru_cache

from docsage.config import settings

# Keys that must never reach Langfuse, even if they end up inside a prompt or error text.
_SECRET = re.compile(r"(sk-or-v1-|sk-lf-|pk-lf-|sk-)[A-Za-z0-9_-]{8,}")


def _mask(*, data, **_):
    if isinstance(data, str):
        return _SECRET.sub("[REDACTED]", data)
    return data


class _NullSpan:
    """Stands in for a Langfuse observation when tracing is off."""

    trace_id = None

    def update(self, **_) -> None:
        pass

    def set_trace_io(self, **_) -> None:
        pass


def enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


@lru_cache(maxsize=1)
def _client():
    if not enabled():
        return None
    # Built from our settings (which read .env) rather than relying on os.environ, so the
    # credentials are always loaded before the client initialises.
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_host,
        environment=settings.langfuse_environment,
        release=settings.langfuse_release or None,  # which code version produced a trace
        mask=_mask,
    )


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs):
    """Open a nested observation. Yields the Langfuse observation (or a null object)."""
    client = _client()
    if client is None:
        yield _NullSpan()
        return
    with client.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
        yield obs


def propagate(**attrs):
    """Set session_id / tags / metadata on every observation created inside the block."""
    if _client() is None:
        return nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(**attrs)


def current_trace_id() -> str | None:
    client = _client()
    return client.get_current_trace_id() if client else None


def score_current_trace(name: str, value: float, comment: str | None = None) -> None:
    client = _client()
    if client:
        client.score_current_trace(name=name, value=value, comment=comment)


def flush() -> None:
    client = _client()
    if client:
        client.flush()


def shutdown() -> None:
    client = _client()
    if client:
        client.shutdown()
