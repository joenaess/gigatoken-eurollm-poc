"""Aggregation and report writers (JSON, Markdown summary, parity report).

All benchmark reporters consume the plain-dict form produced by
:meth:`poc.bench.BenchmarkResult.to_dict`, so reports can be regenerated from
``benchmark_results.json`` without rerunning the benchmark.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tabulate import tabulate

from poc.bench import hardware_summary

OVERALL = "all"
DOMAIN_ORDER = ("prose", "code", "chat", "math")


def write_json(results: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def runs_for(results: Mapping[str, Any], payload: str) -> list[dict[str, Any]]:
    """Runs for one payload, in engine order of first appearance."""
    return [r for r in results["runs"] if r["payload"] == payload]


def domains(results: Mapping[str, Any]) -> list[str]:
    present = {r["payload"] for r in results["runs"]} - {OVERALL}
    ordered = [d for d in DOMAIN_ORDER if d in present]
    return ordered + sorted(present - set(ordered))


def engine_labels(results: Mapping[str, Any]) -> dict[str, str]:
    return {r["engine"]: r["label"] for r in results["runs"]}


def overall_rows(results: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for r in runs_for(results, OVERALL):
        rows.append(
            [
                r["label"],
                f"{r['min_s']:.3f}",
                f"{r['mean_s']:.3f} ± {r['std_s']:.3f}",
                f"{r['mb_per_s']:.1f}",
                f"{r['mtok_per_s']:.2f}",
                f"{r['speedup_vs_baseline']:.2f}×",
            ]
        )
    return rows


def domain_rows(results: Mapping[str, Any]) -> list[list[Any]]:
    labels = engine_labels(results)
    rows = []
    for engine, label in labels.items():
        row: list[Any] = [label]
        for d in domains(results):
            r = next(x for x in runs_for(results, d) if x["engine"] == engine)
            row.append(f"{r['mb_per_s']:.1f} ({r['speedup_vs_baseline']:.2f}×)")
        rows.append(row)
    return rows


def summary_markdown(results: Mapping[str, Any]) -> str:
    env = results["environment"]
    cfg = results["config"]
    overall = cfg["payloads"][OVERALL]
    total_tokens = runs_for(results, OVERALL)[0]["ntokens"]
    baseline_label = engine_labels(results)[results["baseline"]]
    pkgs = env.get("packages", {})

    lines = [
        "## Benchmark summary",
        "",
        f"Tokenizer: `{results['tokenizer']}` · Corpus: {overall['bytes'] / 1e6:.1f} MB, "
        f"{overall['documents']:,} documents, {total_tokens:,} tokens · "
        f"Warmup: {cfg['warmup']} · Timed iterations: {cfg['iterations']} (best time reported)",
        "",
        f"Hardware: {hardware_summary(env)} · Python {env.get('python')} · "
        + ", ".join(f"{k} {v}" for k, v in pkgs.items() if v),
        "",
        "### Overall",
        "",
        tabulate(
            overall_rows(results),
            headers=["Engine", "Best (s)", "Mean ± std (s)", "MB/s", "Mtok/s", "Speedup"],
            tablefmt="github",
            disable_numparse=True,
        ),
        "",
        f"Speedup is relative to **{baseline_label}** (best-time ratio).",
        "",
        "### Per domain — MB/s (speedup)",
        "",
        tabulate(
            domain_rows(results),
            headers=["Engine", *domains(results)],
            tablefmt="github",
            disable_numparse=True,
        ),
        "",
    ]
    return "\n".join(lines)


def write_summary(results: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_markdown(results), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- parity


OUTCOME_COLUMNS = ("passed", "failed", "xfailed", "xpassed", "skipped")


def parity_markdown(
    outcomes: Mapping[str, Mapping[str, int]],
    *,
    inventory: Any | None = None,
    encoders: Iterable[str] = (),
    command: str = "",
    timestamp: str = "",
) -> str:
    """Render per-category parity outcomes (and the derived token inventory) as Markdown."""
    rows = []
    totals = dict.fromkeys(OUTCOME_COLUMNS, 0)
    for cat, counts in outcomes.items():
        n = sum(counts.get(c, 0) for c in OUTCOME_COLUMNS)
        rows.append([cat, n, *(counts.get(c, 0) for c in OUTCOME_COLUMNS)])
        for c in OUTCOME_COLUMNS:
            totals[c] += counts.get(c, 0)
    rows.append(["**total**", sum(totals.values()), *totals.values()])
    ok = totals["failed"] == 0 and totals["xpassed"] == 0
    lines = [
        "# Parity report",
        "",
        f"**Status: {'PASS' if ok else 'FAIL'}** — every case asserts identical token IDs across "
        + ", ".join(f"`{e}`" for e in encoders)
        + ".",
        "",
    ]
    if timestamp or command:
        lines += [f"Generated {timestamp} by `{command}`.".replace("  ", " "), ""]
    lines += [
        tabulate(
            rows,
            headers=["Category", "Cases", *OUTCOME_COLUMNS],
            tablefmt="github",
            disable_numparse=True,
        ),
        "",
    ]
    if totals["xfailed"]:
        lines += [
            "`xfailed` cases are strict expected failures: raw SentencePiece does not match "
            "CONTROL/UNKNOWN pieces (`<unk>`, `<bos>`, `<eos>`, `<pad>`) written as plain text, "
            "while HF and both gigatoken paths map them to their IDs. The remaining paths are "
            "asserted equal for those tokens in separate passing tests.",
            "",
        ]
    if inventory is not None:
        from poc.vocab import RELEASE_NOTES

        s = inventory.summary()
        lines += [
            "## Token inventory (derived from the tokenizer)",
            "",
            tabulate(
                [
                    ["Vocabulary size", s["vocab_size"], ""],
                    [
                        "Added tokens (`added_tokens`)",
                        s["added_tokens"],
                        f"~{RELEASE_NOTES['special_tokens']}",
                    ],
                    ["  of which `special: true`", s["added_tokens_special"], ""],
                    ["SentencePiece USER_DEFINED pieces", s["sp_user_defined"], ""],
                    ["SentencePiece CONTROL pieces", s["sp_control"], ""],
                    ["SentencePiece BYTE pieces", s["sp_byte"], ""],
                    ["Whitespace-only added tokens", s["whitespace_added_tokens"], ""],
                    ["Whitespace-only BPE pieces", s["whitespace_vocab_pieces"], ""],
                    ["Whitespace tokens total", s["whitespace_total"], ""],
                    [
                        "  added + literal-space pieces",
                        s["whitespace_added_tokens"] + s["whitespace_literal_space_pieces"],
                        f"~{RELEASE_NOTES['whitespace_tokens']}",
                    ],
                    [
                        "  reachable when encoding text",
                        s["whitespace_reachable"],
                        "",
                    ],
                ],
                headers=["Quantity", "Derived", "Release notes"],
                tablefmt="github",
                disable_numparse=True,
            ),
            "",
            f"The {s['whitespace_literal_space_pieces']} literal-space pieces (runs of U+0020, "
            "e.g. 2–32 spaces) are never emitted: spaces are escaped to `▁` before BPE, so "
            "indentation is encoded with `▁`-run pieces instead. Every encoding path agrees on "
            "this (see `test_literal_space_pieces_are_never_emitted`).",
            "",
        ]
    return "\n".join(lines)


def write_parity_report(path: Path, outcomes: Mapping[str, Mapping[str, int]], **kw: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(parity_markdown(outcomes, **kw), encoding="utf-8")
    return path
