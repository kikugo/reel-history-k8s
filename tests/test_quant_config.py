import re

from quant import config


def test_revision_is_pinned_to_a_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", config.MODEL_REVISION)


def test_both_recipes_leave_lm_head_unquantized():
    assert "lm_head" in config.RECIPES["awq"]["quant"]["ignore"]
    assert "lm_head" in config.RECIPES["gptq"]["gptq"]["ignore"]


def test_both_recipes_are_4bit_weights_16bit_activations():
    assert config.RECIPES["awq"]["quant"]["scheme"].startswith("W4A16")
    assert config.RECIPES["gptq"]["gptq"]["scheme"].startswith("W4A16")


def test_calibration_is_seeded_and_within_guide_range():
    assert config.CALIB["seed"] == config.SEED
    assert 256 <= config.CALIB["num_samples"] <= 512
    # the split must hold at least as many rows as we ask for
    n = int(re.search(r"\[:(\d+)\]", config.CALIB["split"]).group(1))
    assert n >= config.CALIB["num_samples"]


def test_smoke_only_shrinks_size():
    differs = {k for k in config.CALIB if config.CALIB[k] != config.SMOKE_CALIB[k]}
    assert differs == {"split", "num_samples", "max_seq_length"}


def test_hf_repos_live_under_my_account():
    assert set(config.HF_REPOS) == {"awq", "gptq", "gguf"}
    assert all(r.startswith("karthikcr/") for r in config.HF_REPOS.values())
