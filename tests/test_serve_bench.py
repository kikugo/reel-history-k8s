from bench.serve_bench import LEVELS, bench_cmd, summarize
from serve.variants import model_path, served_name


def _opt(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_command_targets_local_server_with_fixed_shape():
    cmd = bench_cmd("http://127.0.0.1:8000", "gptq", 4, 32, 512, 256, "/tmp/b")
    assert cmd[:3] == ["vllm", "bench", "serve"]
    assert _opt(cmd, "--base-url") == "http://127.0.0.1:8000"
    assert _opt(cmd, "--model") == served_name("gptq")
    assert _opt(cmd, "--tokenizer") == model_path("gptq")
    assert _opt(cmd, "--max-concurrency") == "4"
    assert _opt(cmd, "--random-input-len") == "512"
    assert _opt(cmd, "--random-output-len") == "256"
    assert "--ignore-eos" in cmd  # every request decodes exactly output_len tokens
    assert _opt(cmd, "--temperature") == "0"
    assert _opt(cmd, "--result-filename") == "c4.json"


def test_levels_cover_the_guide_concurrencies():
    assert [c for c, _ in LEVELS] == [1, 4, 16]


def test_summary_keeps_only_aggregate_metrics():
    raw = {"request_throughput": 1.5, "p95_ttft_ms": 80.0, "ttfts": [0.1, 0.2], "generated_texts": ["x"]}
    assert summarize(raw) == {"request_throughput": 1.5, "p95_ttft_ms": 80.0}


def test_llama_bench_rows_split_into_prompt_and_generation():
    from bench.gguf_bench import summarize as gguf_summary

    rows = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 50.5, "stddev_ts": 1.0},
            {"n_prompt": 0, "n_gen": 256, "avg_ts": 9.5, "stddev_ts": 0.2}]
    assert gguf_summary(rows) == {
        "prompt": {"tokens": 512, "avg_tokens_per_s": 50.5, "stddev_tokens_per_s": 1.0},
        "generation": {"tokens": 256, "avg_tokens_per_s": 9.5, "stddev_tokens_per_s": 0.2},
    }
