import pytest

from quant import config
from serve.variants import VARIANTS, app_name, build_vllm_cmd, model_path, served_name


def test_quantized_variants_point_at_phase1_outputs():
    assert model_path("awq") == f"{config.MODELS_DIR}/{config.OUTPUT_NAMES['awq']}"
    assert model_path("gptq") == f"{config.MODELS_DIR}/{config.OUTPUT_NAMES['gptq']}"


def test_bf16_uses_the_pinned_snapshot():
    path = model_path("bf16")
    assert path.startswith(config.HF_HOME)
    assert path.endswith(f"/snapshots/{config.MODEL_REVISION}")
    assert "models--Qwen--Qwen3-8B" in path


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        model_path("gguf")


def test_names_are_unique_per_variant():
    assert len({served_name(v) for v in VARIANTS}) == len(VARIANTS)
    assert len({app_name(v) for v in VARIANTS}) == len(VARIANTS)


def test_command_requires_key_and_listens_externally():
    cmd = build_vllm_cmd("awq", "k3y")
    assert cmd[:3] == ["vllm", "serve", model_path("awq")]
    assert cmd[cmd.index("--api-key") + 1] == "k3y"
    assert cmd[cmd.index("--host") + 1] == "0.0.0.0"
    assert cmd[cmd.index("--port") + 1] == "8000"


def test_every_variant_gets_the_same_memory_budget():
    # Phase 3 compares KV-cache capacity, so these must not differ between variants.
    def opt(cmd, flag):
        return cmd[cmd.index(flag) + 1]

    cmds = [build_vllm_cmd(v, "k") for v in VARIANTS]
    for flag in ("--max-model-len", "--gpu-memory-utilization", "--seed"):
        assert len({opt(c, flag) for c in cmds}) == 1
