from bench.common import parse_vllm_log

# Lines copied from a real vLLM 0.32 startup log on an L4 (Phase 1 load check, AWQ).
LOG = """\
(EngineCore pid=46) INFO 10-25 20:31:35 [model_runner.py:424] Model loading took 5.71 GiB memory and 12.336325 seconds
(EngineCore pid=46) INFO 10-25 20:33:04 [gpu_worker.py:759] Available KV cache memory: 11.04 GiB
(EngineCore pid=46) INFO 10-25 20:33:04 [kv_cache_utils.py:2327] GPU KV cache size: 80,384 tokens, \
Maximum concurrency for 4,096 tokens per request: 19.62x
"""


def test_parses_weights_and_kv_cache():
    assert parse_vllm_log(LOG) == {
        "weight_gib": 5.71,
        "kv_cache_gib": 11.04,
        "kv_cache_tokens": 80384,
        "max_concurrency": 19.62,
        "max_concurrency_at_tokens": 4096,
    }


def test_missing_lines_give_none():
    assert set(parse_vllm_log("nothing here").values()) == {None}


def test_last_value_wins_after_a_restart():
    second = LOG.replace("80,384", "79,000")
    assert parse_vllm_log(LOG + second)["kv_cache_tokens"] == 79000
