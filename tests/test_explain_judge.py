import pytest

from bench.explain import per_thousand, render_qwen3
from bench.judge import combine, judge_message, parse_verdict, tally
from quant.verify import PROMPT, PROMPT_RENDERED


def test_render_matches_the_template_checked_in_phase1():
    # quant/verify.py confirmed PROMPT_RENDERED against the real Qwen3 tokenizer.
    assert render_qwen3(PROMPT) == PROMPT_RENDERED


def test_cost_per_thousand():
    assert per_thousand(0.05, 50) == 1.0


@pytest.mark.parametrize("first,second,expected", [
    ("A", "B", "win"),    # candidate preferred in both orders
    ("B", "A", "loss"),   # reference preferred in both orders
    ("A", "A", "tie"),    # position bias: always picks the first answer
    ("B", "B", "tie"),
    ("tie", "B", "tie"),
])
def test_combine_requires_agreement_across_orders(first, second, expected):
    assert combine(first, second) == expected


def test_tally_counts_every_outcome():
    assert tally(["win", "tie", "tie", "loss"]) == {"win": 1, "tie": 2, "loss": 1}


def test_parse_verdict_accepts_wrapped_json():
    assert parse_verdict('Sure: {"winner": "B", "reason": "more specific"}') == "B"


def test_parse_verdict_rejects_unknown_winner():
    with pytest.raises(ValueError):
        parse_verdict('{"winner": "C"}')


def test_judge_message_keeps_both_answers_and_the_request():
    msg = judge_message("REQ", "first answer", "second answer")
    assert "REQ" in msg and "first answer" in msg and "second answer" in msg
    assert msg.index("first answer") < msg.index("second answer")
    assert '"winner"' in msg  # braces in the JSON instruction survived str.format


def test_retry_delay_prefers_retry_after_and_caps():
    from bench.judge import retry_delay

    assert retry_delay("7", 0) == 7.5
    assert retry_delay("600", 0) == 60.5
    assert retry_delay(None, 3) == 8.0
    assert retry_delay("soon", 10) == 60.0
