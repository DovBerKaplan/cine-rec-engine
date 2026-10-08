"""Serving metrics (RFC M3): counters, snapshot math, Prometheus
rendering, and the /metrics endpoint shape."""

from __future__ import annotations

import pytest

from cine_rec_engine import metrics


def setup_function(_fn):
    metrics.reset()


def test_latency_histogram_buckets():
    for ms in (3, 7, 40, 900):
        metrics.observe_request("/similar", ms)
    s = metrics.snapshot()["requests"]["/similar"]
    assert s["count"] == 4
    # 3→5ms bucket, 7→10, 40→50, 900→1000
    assert s["bucket_counts"][0] == 1
    assert s["bucket_counts"][1] == 1
    assert s["bucket_counts"][3] == 1
    assert s["bucket_counts"][7] == 1
    assert s["mean_ms"] == 237.5


def test_cache_hit_rate_and_none_before_any_traffic():
    assert metrics.snapshot()["cache"]["hit_rate"] is None
    metrics.observe_cache("miss")
    metrics.observe_cache("hit")
    metrics.observe_cache("hit")
    c = metrics.snapshot()["cache"]
    assert c["hit"] == 2 and c["miss"] == 1 and c["store"] == 0
    assert c["hit_rate"] == pytest.approx(2 / 3, abs=0.001)


def test_channel_coverage_counts_failed_as_total_not_contributed():
    metrics.observe_channel("genre", 12)
    metrics.observe_channel("genre", 0)
    ch = metrics.snapshot()["channels"]["genre"]
    assert ch == {"coverage": 0.5, "contributed": 1, "total": 2}


def test_prometheus_rendering_has_the_three_families():
    metrics.observe_request("/for-user/42", 120.0)
    metrics.observe_cache("hit")
    metrics.observe_channel("KNN", 5)
    text = metrics.render_prometheus()
    assert "cine_rec_requests_total{path=\"/for-user/42\"} 1" in text
    assert "cine_rec_cache_hit_rate 1.0" in text
    assert 'cine_rec_channel_coverage{channel="KNN"} 1.0' in text
    assert 'le="+Inf"' in text


def test_metrics_endpoint_serves_prometheus_text():
    from fastapi.testclient import TestClient

    from cine_rec_engine.serve import create_app

    metrics.observe_cache("miss")
    app = create_app(service=object())  # /metrics never touches the engine
    with TestClient(app) as client:
        r = client.get("/metrics")
    assert r.status_code == 200
    assert "cine_rec_cache_misses_total 1" in r.text
