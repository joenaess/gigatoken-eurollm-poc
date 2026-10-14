from pathlib import Path

import pytest

from poc.bench import BenchmarkResult, RunStats
from poc.plots import PLOT_FILES, engine_colors, write_all_plots
from poc.reporter import (
    domains,
    load_json,
    parity_markdown,
    summary_markdown,
    write_json,
    write_parity_report,
    write_summary,
)

ENGINES = {"hf_fast": ("HF Fast", 4.0), "gt_compat": ("GT compat", 2.0), "gt_native": ("GT", 1.0)}
PAYLOADS = {
    "all": 8_000_000,
    "prose": 5_600_000,
    "code": 1_200_000,
    "chat": 800_000,
    "math": 400_000,
}
ENV = {
    "cpu_model": "Test CPU",
    "physical_cores": 4,
    "logical_threads": 8,
    "ram_gb": 16.0,
    "python": "3.12.0",
    "packages": {"gigatoken": "0.0.0", "tokenizers": "0.0.1"},
}


@pytest.fixture(scope="module")
def results() -> dict:
    stats = [
        RunStats(
            engine=e,
            label=label,
            payload=p,
            nbytes=nbytes,
            ndocs=10,
            ntokens=nbytes // 4,
            times_s=[secs * nbytes / 8e6, secs * nbytes / 8e6 * 1.1, secs * nbytes / 8e6 * 1.2],
        )
        for p, nbytes in PAYLOADS.items()
        for e, (label, secs) in ENGINES.items()
    ]
    config = {
        "iterations": 3,
        "warmup": 1,
        "payloads": {p: {"bytes": b, "documents": 10} for p, b in PAYLOADS.items()},
    }
    return BenchmarkResult(stats, "hf_fast", config, ENV).to_dict()


def test_json_roundtrip(tmp_path: Path, results):
    path = write_json(results, tmp_path / "r" / "benchmark_results.json")
    assert load_json(path) == results


def test_domains_ordered(results):
    assert domains(results) == ["prose", "code", "chat", "math"]


def test_summary_markdown_contents(results):
    md = summary_markdown(results)
    assert "## Benchmark summary" in md
    assert "Test CPU (4 cores / 8 threads, 16.0 GB RAM)" in md
    assert "| Engine" in md and "| HF Fast" in md
    assert "4.00×" in md  # GT native vs HF Fast
    assert "2.00×" in md
    assert "8.0 MB" in md and "2,000,000 tokens" in md
    for d in ("prose", "code", "chat", "math"):
        assert d in md


def test_write_summary(tmp_path, results):
    path = write_summary(results, tmp_path / "s.md")
    assert path.read_text(encoding="utf-8") == summary_markdown(results)


def test_engine_colors_are_stable(results):
    colors = engine_colors(results)
    assert list(colors) == list(ENGINES)
    assert len(set(colors.values())) == len(ENGINES)


def test_write_all_plots(tmp_path, results):
    paths = write_all_plots(results, tmp_path)
    assert set(paths) == set(PLOT_FILES)
    for key, path in paths.items():
        assert path.name == PLOT_FILES[key]
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG") and len(data) > 10_000


def test_parity_markdown_counts_and_status():
    outcomes = {
        "whitespace": {"passed": 10},
        "specials": {"passed": 5, "xfailed": 2},
        "scripts": {"passed": 3, "failed": 1},
    }
    md = parity_markdown(outcomes, encoders=["a", "b"], command="pytest", timestamp="now")
    assert "**Status: FAIL**" in md
    assert "| **total**  | 21" in md
    assert "`a`, `b`" in md
    assert "CONTROL/UNKNOWN" in md


def test_parity_markdown_pass():
    md = parity_markdown({"whitespace": {"passed": 3}})
    assert "**Status: PASS**" in md
    assert "CONTROL/UNKNOWN" not in md


def test_parity_report_with_inventory(tmp_path):
    class Inv:
        def summary(self):
            return {
                "vocab_size": 100,
                "added_tokens": 7,
                "added_tokens_special": 5,
                "sp_user_defined": 4,
                "sp_control": 3,
                "sp_byte": 2,
                "whitespace_added_tokens": 1,
                "whitespace_vocab_pieces": 2,
                "whitespace_total": 3,
                "whitespace_literal_space_pieces": 1,
                "whitespace_reachable": 2,
            }

    path = write_parity_report(tmp_path / "p.md", {"x": {"passed": 1}}, inventory=Inv())
    md = path.read_text(encoding="utf-8")
    assert "Token inventory" in md
    assert "| Whitespace tokens total" in md and "~45" in md and "~204" in md
