"""
Live API connectivity tests.
Each test makes one real API call to verify keys, tiers, and endpoints.
"""
import pytest
import cohere
import anthropic


# ── Anthropic ──────────────────────────────────────────────────────────────────

def test_anthropic_api_responds(settings):
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": "Reply with OK"}],
    )
    assert msg.content[0].text.strip(), "Anthropic API returned empty response"


def test_anthropic_haiku_model_accessible(settings):
    """Haiku is the eval model — verify it's available."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model=settings.eval_model,
        max_tokens=5,
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert msg.stop_reason in ("end_turn", "max_tokens")


def test_anthropic_prompt_caching_header(settings):
    """Verify cache_control is accepted (not rejected as invalid param)."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        system=[{
            "type": "text",
            "text": "You are a helpful assistant.",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert msg.content, "Prompt caching request was rejected"


# ── Cohere ────────────────────────────────────────────────────────────────────

def test_cohere_embed_production_key(settings):
    """
    Trial keys return x-trial-endpoint-call-limit header.
    Production keys do not. Fail if we detect a trial key.
    """
    import httpx
    resp = httpx.post(
        "https://api.cohere.com/v2/embed",
        headers={"Authorization": f"Bearer {settings.cohere_api_key}", "Content-Type": "application/json"},
        json={"texts": ["test"], "model": "embed-english-v3.0", "input_type": "search_query", "embedding_types": ["float"]},
        timeout=15,
    )
    assert resp.status_code == 200, f"Cohere embed failed: {resp.status_code} {resp.text[:200]}"
    trial_limit = resp.headers.get("x-trial-endpoint-call-limit")
    assert trial_limit is None, (
        f"COHERE_API_KEY is a TRIAL key (limit: {trial_limit}). "
        "Switch to a production key before running eval."
    )


def test_cohere_embed_returns_vectors(settings):
    co = cohere.ClientV2(api_key=settings.cohere_api_key)
    resp = co.embed(
        texts=["waiting period for maternity"],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    )
    embeddings = resp.embeddings.float_
    assert embeddings and len(embeddings[0]) == 1024, \
        f"Expected 1024-dim embeddings, got {len(embeddings[0]) if embeddings else 'none'}"


def test_cohere_rerank_returns_results(settings):
    co = cohere.ClientV2(api_key=settings.cohere_api_key)
    resp = co.rerank(
        model="rerank-english-v3.0",
        query="maternity waiting period",
        documents=["24-month wait for maternity", "room rent capped at 1%", "PED exclusion applies"],
        top_n=2,
    )
    assert len(resp.results) == 2
    assert resp.results[0].relevance_score > 0


# ── Pinecone ──────────────────────────────────────────────────────────────────

def test_pinecone_index_exists(pinecone_index, settings):
    stats = pinecone_index.describe_index_stats()
    assert stats["dimension"] == 1024, f"Expected 1024 dims, got {stats['dimension']}"
    assert stats["metric"] == "cosine", f"Expected cosine metric, got {stats['metric']}"


def test_pinecone_policy_namespace_has_vectors(pinecone_index, policy_id):
    stats = pinecone_index.describe_index_stats()
    ns = stats.get("namespaces", {})
    assert policy_id in ns, \
        f"Namespace '{policy_id}' not found in Pinecone. Run ingestion first."
    count = ns[policy_id]["vector_count"]
    assert count > 100, \
        f"Only {count} vectors in '{policy_id}' namespace — ingestion may be incomplete"


def test_pinecone_query_returns_results(pinecone_index, settings, policy_id):
    import cohere as co_lib
    co = co_lib.ClientV2(api_key=settings.cohere_api_key)
    emb = co.embed(
        texts=["maternity waiting period"],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    ).embeddings.float_[0]

    results = pinecone_index.query(vector=emb, top_k=5, namespace=policy_id, include_metadata=True)
    assert len(results.matches) > 0, "Pinecone query returned no results"
    assert results.matches[0].score > 0.3, \
        f"Top result score too low: {results.matches[0].score:.3f} — index may be empty"
