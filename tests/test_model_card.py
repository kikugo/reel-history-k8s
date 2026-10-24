import json
import re

import pytest

from quant import config
from quant.model_card import load_record, render

# Odd values on purpose, so a hand-typed number in the template can't hide among them.
RUN = {
    "base_model": config.MODEL_ID,
    "base_revision": config.MODEL_REVISION,
    "finished_utc": "2099-01-02T03:04:05+00:00",
    "timings_s": {"load_s": 11.1, "calibrate_s": 2222.2, "save_s": 33.3},
    "wall_clock_s": 2266.6,
    "hardware": {"requested_gpu": "A100", "gpu_name": "NVIDIA A100-SXM4-80GB", "cpu_cores": 8, "mem_gib": 64},
    "cost_usd": 1.7171,
    "price_source": {"url": "https://modal.com/pricing", "date": "2099-12-31"},
}

AWQ = {
    **RUN,
    "method": "awq",
    "smoke": False,
    "calibration": {**config.CALIB, "num_samples": 513, "max_seq_length": 2047, "seed": 41},
    "recipe": config.RECIPES["awq"],
    "versions": {"llmcompressor": "9.9.9", "torch": "8.8.8"},
}

GGUF = {
    **RUN,
    "method": "gguf",
    "smoke": False,
    "hardware": {**RUN["hardware"], "requested_gpu": None, "gpu_name": "none"},
    "quant_type": config.GGUF_QUANT,
    "llama_cpp_tag": config.LLAMA_CPP_TAG,
    "imatrix": False,
    "file_bytes": {"f16": 16_000_000_001, "quant": 5_000_000_003},
}

# Names that legitimately contain digits.
IDENTIFIERS = [
    config.MODEL_ID, config.MODEL_REVISION, config.MODEL_NAME, config.GGUF_QUANT, config.LLAMA_CPP_TAG,
    "apache-2.0", "4-bit", "BF16", "W4A16_ASYM", "W4A16", "w4a16", "qwen3", "Qwen3", "f16", "--outtype f16", "llama.cpp",
    *config.OUTPUT_NAMES.values(), *config.HF_REPOS.values(),
]


def _leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v)
    else:
        yield str(obj)


def _strip_known(text: str, rec: dict) -> str:
    known = list(_leaves(rec)) + [json.dumps(rec.get("recipe", ""))] + IDENTIFIERS
    for s in sorted(known, key=len, reverse=True):
        text = text.replace(s, "")
    return text


@pytest.mark.parametrize("rec", [AWQ, GGUF], ids=["awq", "gguf"])
def test_every_number_in_the_card_comes_from_the_record(rec):
    leftover = _strip_known(render(rec), rec)
    assert not re.findall(r"\d", leftover), leftover


def test_awq_card_shows_calibration_and_cost():
    card = render(AWQ)
    for value in ("513", "2047", "41", "1.7171", "2266.6", "mlabonne/open-perfectblend", "lm_head"):
        assert value in card
    assert card.startswith("---\nbase_model: Qwen/Qwen3-8B\n")


def test_gguf_card_names_the_file():
    assert "Qwen3-8B-Q4_K_M.gguf" in render(GGUF)


def test_card_never_says_production():
    assert "production" not in render(AWQ).lower() + render(GGUF).lower()


def test_smoke_records_are_refused(tmp_path):
    (tmp_path / "awq.json").write_text(json.dumps({**AWQ, "smoke": True}))
    with pytest.raises(ValueError):
        load_record("awq", tmp_path)
