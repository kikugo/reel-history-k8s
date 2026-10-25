import pytest

from quant.config import CPU_PRICE_PER_CORE_SEC, GPU_PRICE_PER_SEC, MEM_PRICE_PER_GIB_SEC, cost_usd


def test_gpu_only():
    assert cost_usd(3600, "A100") == round(3600 * GPU_PRICE_PER_SEC["A100"], 4)


def test_gpu_cpu_and_memory_add_up():
    expected = 100 * (GPU_PRICE_PER_SEC["L4"] + 8 * CPU_PRICE_PER_CORE_SEC + 64 * MEM_PRICE_PER_GIB_SEC)
    assert cost_usd(100, "L4", cpu_cores=8, mem_gib=64) == round(expected, 4)


def test_cpu_only_job():
    assert cost_usd(60, None, cpu_cores=1) == round(60 * CPU_PRICE_PER_CORE_SEC, 4)


def test_a100_request_is_billed_at_40gb_price():
    assert GPU_PRICE_PER_SEC["A100"] == GPU_PRICE_PER_SEC["A100-40GB"]


def test_unknown_gpu_raises():
    with pytest.raises(KeyError):
        cost_usd(10, "T9000")


def test_negative_time_raises():
    with pytest.raises(ValueError):
        cost_usd(-1, "L4")
