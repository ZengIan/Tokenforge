"""Loads the GPU database from gpus.yaml and exposes it as GpuSpec objects."""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

from ..models.schemas import GpuSpec

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GPU_YAML = DATA_DIR / "gpus.yaml"


@functools.lru_cache(maxsize=1)
def load_gpus() -> list[GpuSpec]:
    with GPU_YAML.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    specs: list[GpuSpec] = []
    for name, body in (raw.get("gpus") or {}).items():
        body = body or {}
        specs.append(
            GpuSpec(
                name=name,
                mem_gb=float(body.get("mem_gb", 0)),
                bw_gbs=float(body.get("bw_gbs", 0)),
                fp16_tflops=float(body.get("fp16_tflops", 0)),
                fp8_tflops=float(body.get("fp8_tflops", 0)),
                nvlink=bool(body.get("nvlink", False)),
                source=str(body.get("source", "estimate")),
                note=str(body.get("note", "")),
            )
        )
    return specs


def reload_gpus() -> list[GpuSpec]:
    load_gpus.cache_clear()
    return load_gpus()
