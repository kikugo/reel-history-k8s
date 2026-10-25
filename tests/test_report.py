import json
import re

import pytest

from bench import report


def _serve(variant, weights, tput1, tput16):
    return {
        "variant": variant, "gpu_name": "NVIDIA L4", "input_len": 512, "output_len": 256,
        "memory": {"weight_gib": weights, "kv_cache_gib": 3.33, "kv_cache_tokens": 22222,
                   "max_concurrency": 5.43, "max_concurrency_at_tokens": 4096},
        "levels": [
            {"concurrency": 1, "output_throughput": tput1, "request_throughput": 0.11, "p50_ttft_ms": 41.0,
             "p95_ttft_ms": 47.0, "p50_tpot_ms": 21.0, "p95_tpot_ms": 23.0},
            {"concurrency": 16, "output_throughput": tput16, "request_throughput": 1.11, "p50_ttft_ms": 141.0,
             "p95_ttft_ms": 247.0, "p50_tpot_ms": 31.0, "p95_tpot_ms": 33.0},
        ],
    }


FIXTURES = {
    "serve_bf16": _serve("bf16", 15.27, 20.0, 200.0),
    "serve_awq": _serve("awq", 5.71, 50.0, 500.0),
    "ppl_awq": {"wikitext2": {"word_perplexity": 12.345}, "frwiki": {"word_perplexity": 9.876}},
    "judge_awq": {"overall": {"win": 7, "tie": 30, "loss": 13}, "n": 50,
                  "judge": {"provider": "groq", "model": "openai/gpt-oss-120b"}},
    "explain_awq": {"hardware": "L4", "concurrency": 16, "generation_s": 33.3, "cost_per_1000_usd": 0.271, "n": 50},
}

README = "# x\n\nintro\n\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n\n## Run it\nkeep me\n"


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "bench"
    d.mkdir()
    for name, rec in FIXTURES.items():
        (d / f"{name}.json").write_text(json.dumps(rec))
    (d / "serve_gptq_smoke.json").write_text(json.dumps(_serve("gptq", 1.0, 1.0, 1.0)))
    return d


def test_ratios_are_computed_not_typed(results_dir):
    text = report.render(report.load(results_dir))
    assert "2.50x at concurrency 1" in text and "2.50x at concurrency 16" in text
    assert "9.56 GiB less weight memory" in text  # 15.27 - 5.71


def test_smoke_files_are_ignored(results_dir):
    assert "GPTQ" not in report.render(report.load(results_dir))


def test_tables_carry_the_fixture_values(results_dir):
    text = report.render(report.load(results_dir))
    for value in ("12.35", "9.88", "7 / 30 / 13", "0.271", "22,222", "500.0"):
        assert value in text


def test_no_results_means_no_numbers():
    text = report.render({})
    assert not re.findall(r"\d", text.replace("results/bench", ""))


def test_only_the_marked_block_changes_and_rerun_is_stable(results_dir, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(README)
    report.main(results_dir, readme)
    once = readme.read_text()
    report.main(results_dir, readme)
    assert readme.read_text() == once
    assert once.startswith("# x\n\nintro\n\n") and once.endswith("## Run it\nkeep me\n")
    assert "old" not in once
    assert (results_dir / "throughput_vs_concurrency.png").exists()


def test_missing_markers_raise():
    with pytest.raises(ValueError):
        report.replace_block("no markers", "x")
