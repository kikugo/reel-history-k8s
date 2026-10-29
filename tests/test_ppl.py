from pathlib import Path

import pytest

from bench.ppl import lm_eval_cmd, pick_metrics
from bench.tasks.frwiki import process_results
from serve.variants import GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, model_path

TASKS = Path(__file__).resolve().parent.parent / "bench" / "tasks"


def _opt(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_uses_vllm_with_the_serving_memory_budget():
    cmd = lm_eval_cmd("awq", smoke=False, out_dir="/tmp/x")
    args = _opt(cmd, "--model_args")
    assert _opt(cmd, "--model") == "vllm"
    assert f"pretrained={model_path('awq')}" in args
    assert f"max_model_len={MAX_MODEL_LEN}" in args
    assert f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION}" in args
    assert _opt(cmd, "--tasks") == "wikitext,frwiki_ppl"
    assert "--limit" not in cmd


def test_smoke_limits_docs_and_uses_small_slice():
    cmd = lm_eval_cmd("bf16", smoke=True, out_dir="/tmp/x")
    assert _opt(cmd, "--limit") == "2"
    assert "frwiki_ppl_smoke" in _opt(cmd, "--tasks")


def test_pick_metrics_reads_lm_eval_keys():
    res = {"results": {"wikitext": {"word_perplexity,none": 9.5, "byte_perplexity,none": 1.5,
                                    "bits_per_byte,none": 0.58, "alias": "wikitext"}}}
    assert pick_metrics(res, "wikitext") == {"word_perplexity": 9.5, "byte_perplexity": 1.5, "bits_per_byte": 0.58}


def test_french_counts_words_and_utf8_bytes():
    out = process_results({"page": "Le café est très bon"}, (-10.0,))
    assert out["word_perplexity"] == (-10.0, 5)
    assert out["byte_perplexity"] == (-10.0, len("Le café est très bon".encode()))


@pytest.mark.parametrize("name", ["frwiki_ppl.yaml", "frwiki_ppl_smoke.yaml"])
def test_task_yaml_points_at_the_volume(name):
    text = (TASKS / name).read_text()
    assert "/vol/data/frwiki_slice" in text
    assert "loglikelihood_rolling" in text
