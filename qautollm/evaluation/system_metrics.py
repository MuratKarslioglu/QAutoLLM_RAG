"""Latency/token/VRAM system metrics recorded alongside every trial."""

from __future__ import annotations

from dataclasses import dataclass

from qautollm.utils.profiling import vram_probe


@dataclass(frozen=True)
class SystemMetrics:
    latency_s: float
    tokens_prompt: int
    tokens_generated: int
    peak_vram_mb: float


def collect_system_metrics(
    latency_s: float, tokens_prompt: int, tokens_generated: int
) -> SystemMetrics:
    vram = vram_probe()
    peak_vram_mb = vram["max_allocated_mb"] if vram.get("available") else 0.0
    return SystemMetrics(
        latency_s=latency_s,
        tokens_prompt=tokens_prompt,
        tokens_generated=tokens_generated,
        peak_vram_mb=peak_vram_mb,
    )
