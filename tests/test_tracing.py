from docsage.observability import tracing


def test_mask_redacts_api_keys_and_leaves_other_text():
    assert tracing._mask(data="key sk-or-v1-abcdef1234567890 here") == "key [REDACTED] here"
    assert tracing._mask(data="pk-lf-12345678abcd") == "[REDACTED]"
    assert tracing._mask(data="Attrition was 13.7%") == "Attrition was 13.7%"
    assert tracing._mask(data={"a": 1}) == {"a": 1}  # non-strings pass through


def test_tracing_is_a_noop_without_keys(monkeypatch):
    monkeypatch.setattr(tracing.settings, "langfuse_public_key", "")
    monkeypatch.setattr(tracing.settings, "langfuse_secret_key", "")
    tracing._client.cache_clear()
    assert not tracing.enabled()
    with (
        tracing.propagate(session_id="s", tags=["x"]),
        tracing.observation("anything", as_type="generation", model="m") as span,
    ):
        span.update(output="x")
        span.set_trace_io(input="q", output="a")
    assert tracing.current_trace_id() is None
    tracing.score_current_trace("citations_valid", 1.0)  # must not raise
    tracing.flush()
    tracing.shutdown()
