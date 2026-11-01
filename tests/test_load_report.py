import importlib.util
import json
from pathlib import Path

import pytest

from load import report

_spec = importlib.util.spec_from_file_location(
    "load_test", Path(__file__).resolve().parent.parent / "scripts" / "load_test.py")
load_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(load_test)
step_passes, sustained = load_test.step_passes, load_test.sustained


def step(rate, p95, passed=True, err=0.0, dropped=0, pods=2):
    return {"offered_rps": rate, "achieved_rps": rate * 0.99, "requests": rate * 60, "p50_ms": p95 / 2,
            "p95_ms": p95, "p99_ms": p95 * 2, "max_ms": p95 * 4, "error_rate": err,
            "dropped_iterations": dropped, "recommender_replicas": pods, "passed": passed}


CRIT = {"p95_ms_below": 200.0, "error_rate_below": 0.01, "dropped_share_at_most": 0.01, "warm_s": 60, "measure_s": 60}


def fixture_data():
    off = {"criteria": CRIT, "docker_cpus": 12, "sustained_rps": 200,
           "steps": [step(100, 3.0), step(200, 9.5), step(400, 450.0, passed=False)]}
    on = {"criteria": CRIT, "docker_cpus": 12, "sustained_rps": 400,
          "steps": [step(100, 3.0), step(200, 8.0, pods=3), step(400, 60.0, pods=5), step(600, 900.0, False, pods=5)]}
    kill = {"rate_rps": 250, "duration_s": 240, "kill_after_s": 90, "pod": "recommender-abc", "seconds_to_full": 12.5,
            "replicas_at_kill": 3, "overall": step(250, 5.0), "failed_total": 4, "failed_30s_after_kill": 4,
            "worst_p95_ms_30s_before": 4.0, "worst_p95_ms_30s_after": 35.0}
    return {"hpa_off": off, "hpa_on": on, "pod_kill": kill}


def test_sustained_is_highest_pass_before_first_failure():
    steps = [step(100, 3), step(200, 5), step(400, 300, passed=False), step(600, 5)]
    assert sustained(steps)["offered_rps"] == 200
    assert sustained([step(100, 300, passed=False)]) is None


def test_step_passes_checks_latency_errors_and_dropped():
    assert step_passes(step(100, 199.9))
    assert not step_passes(step(100, 200.0))
    assert not step_passes(step(100, 10, err=0.02))
    assert not step_passes(step(100, 10, dropped=61))  # more than 1% of 100 req/s x 60 s


def test_render_takes_every_number_from_the_results():
    text = report.render(fixture_data())
    assert "| HPA off (2 replicas) | 200 | 198 | 9.5 | 2 |" in text
    assert "| HPA on (2-5 replicas) | 400 | 396 | 60.0 | 5 |" in text
    assert "| 400 | 450.0 | **fail** | 60.0 | 5 | pass |" in text
    assert "| 600 |  |  | 900.0 | 5 | **fail** |" in text
    assert "`recommender-abc` was deleted 90 s in" in text and "4 of 15,000 requests failed" in text


def test_update_readme_replaces_only_the_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.START}\nold\n{report.END}\noutro\n")
    report.update_readme("new block", readme)
    assert readme.read_text() == f"intro\n{report.START}\nnew block\n{report.END}\noutro\n"


def test_missing_results_say_what_to_run(tmp_path):
    (tmp_path / "hpa_off.json").write_text(json.dumps({}))
    with pytest.raises(SystemExit, match="run scripts/load_test.py"):
        report.load(tmp_path)


def test_plot_writes_a_png(tmp_path):
    out = tmp_path / "chart.png"
    report.plot(fixture_data(), out)
    assert out.read_bytes()[:4] == b"\x89PNG"
