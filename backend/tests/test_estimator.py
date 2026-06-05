"""Sanity tests for the estimation engine.

Run from backend/:  python -m pytest   (or: python tests/test_estimator.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import (  # noqa: E402
    EstimateRequest,
    GpuGroup,
    InferenceConfig,
    ModelSpec,
)
from app.services.estimator import estimate  # noqa: E402
from app.services.gpu_db import load_gpus  # noqa: E402

GPUS = {g.name: g for g in load_gpus()}


def _req(gpu: str, count: int, **inf) -> EstimateRequest:
    return EstimateRequest(
        model=ModelSpec(
            model_id="qwen2-72b",
            params_b=72,
            hidden_size=8192,
            num_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            vocab_size=152064,
        ),
        gpus=[GpuGroup(spec=GPUS[gpu], count=count)],
        inference=InferenceConfig(**inf),
    )


def test_gpu_db_loads():
    assert len(GPUS) >= 8
    assert "NVIDIA H20-3e (141G)" in GPUS


def test_weights_scale_with_quant():
    fp16 = estimate(_req("NVIDIA H20-3e (141G)", 8, quantization="none", dtype="float16"))
    awq = estimate(_req("NVIDIA H20-3e (141G)", 8, quantization="awq"))
    # AWQ 4bit weights ~ 1/4 of FP16 (0.5 vs 2 bytes), allowing group overhead
    assert awq.memory.weights_gb < fp16.memory.weights_gb * 0.3


def test_higher_bandwidth_improves_decode():
    # H20-3e has higher bandwidth than A100 -> lower TPOT (faster decode)
    h20 = estimate(_req("NVIDIA H20-3e (141G)", 8, max_num_seqs=16))
    a100 = estimate(_req("NVIDIA A100 SXM (80G)", 8, max_num_seqs=16))
    assert h20.tpot_ms < a100.tpot_ms


def test_concurrency_increases_tps():
    low = estimate(_req("NVIDIA H100 SXM (80G)", 8, max_num_seqs=1))
    high = estimate(_req("NVIDIA H100 SXM (80G)", 8, max_num_seqs=32))
    assert high.tps > low.tps


def test_oversized_model_does_not_fit():
    # 72B FP16 weights (~134GB) on a single small card cannot fit
    r = estimate(_req("NVIDIA A100 SXM (80G)", 1, quantization="none", dtype="float16", max_num_seqs=1))
    assert r.fits is False


def test_fp8_warns_on_ampere():
    r = estimate(_req("NVIDIA A100 SXM (80G)", 8, quantization="fp8"))
    assert any("FP8" in w for w in r.warnings)


def test_tps_range_brackets_mid():
    r = estimate(_req("NVIDIA H20-3e (141G)", 2, max_num_seqs=16))
    assert r.single_tps_low <= r.single_tps <= r.single_tps_high
    assert r.tps_low <= r.tps <= r.tps_high


def test_enforce_eager_lowers_single_tps():
    on = estimate(_req("NVIDIA H20-3e (141G)", 2, max_num_seqs=16, enforce_eager=True))
    off = estimate(_req("NVIDIA H20-3e (141G)", 2, max_num_seqs=16, enforce_eager=False))
    # 关闭 CUDA Graph(eager) 单请求更慢
    assert on.single_tps < off.single_tps


def test_mla_kv_far_smaller_than_mha():
    """MLA(低秩 latent KV) 应远小于把同模型当 MHA 算的 KV。"""
    g = GPUS["NVIDIA H100 SXM (80G)"]
    base = dict(
        params_b=210, hidden_size=6144, num_layers=78,
        num_attention_heads=64, num_key_value_heads=64, vocab_size=152064,
    )
    inf = dict(max_model_len=131072, max_num_seqs=8)
    mha = estimate(EstimateRequest(
        model=ModelSpec(**base, attn_type="MHA"),
        gpus=[GpuGroup(spec=g, count=8)], inference=InferenceConfig(**inf)))
    mla = estimate(EstimateRequest(
        model=ModelSpec(**base, attn_type="MLA", mla_kv_dim=576),
        gpus=[GpuGroup(spec=g, count=8)], inference=InferenceConfig(**inf)))
    # MLA KV 应至少小 10 倍, 且可容纳并发更多
    assert mla.memory.kv_cache_gb < mha.memory.kv_cache_gb / 10
    assert mla.max_fit_seqs > mha.max_fit_seqs


def test_max_fit_seqs_reported():
    # 庞大并发应触发 max_fit_seqs < max_num_seqs
    r = estimate(_req("NVIDIA A100 SXM (80G)", 8, max_num_seqs=2048, max_model_len=131072))
    assert r.max_fit_seqs >= 0
    assert r.max_fit_seqs < 2048


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
