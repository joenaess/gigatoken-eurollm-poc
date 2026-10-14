"""Result graphs rendered with matplotlib's non-interactive Agg backend.

Every chart consumes the dict produced by :meth:`poc.bench.BenchmarkResult.to_dict`. Each
engine keeps one fixed colour across all charts; values are labelled on every bar so colour
never carries identity alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from poc.bench import hardware_summary  # noqa: E402
from poc.reporter import OVERALL, domains, engine_labels, runs_for  # noqa: E402

DPI = 150

# Validated categorical palette (fixed order, light surface): blue, orange, aqua, yellow, ...
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

PLOT_FILES = {
    "mbs": "throughput_mbs.png",
    "mtoks": "throughput_mtoks.png",
    "speedup": "speedup.png",
    "domain": "throughput_by_domain.png",
}


def engine_colors(results: Mapping[str, Any]) -> dict[str, str]:
    """Colour follows the engine (order of first appearance), never its rank."""
    return {e: SERIES_COLORS[i % len(SERIES_COLORS)] for i, e in enumerate(engine_labels(results))}


def _style(ax: Axes, value_axis: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis=value_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _footnote(fig: Figure, results: Mapping[str, Any]) -> None:
    cfg = results["config"]
    mb = cfg["payloads"][OVERALL]["bytes"] / 1e6
    text = (
        f"{hardware_summary(results['environment'])} · {mb:.0f} MB corpus · "
        f"best of {cfg['iterations']} timed runs after {cfg['warmup']} warmup · "
        f"tokenizer {results['tokenizer']}"
    )
    fig.text(0.01, -0.05, text, ha="left", va="top", fontsize=7.5, color=INK_MUTED)


def _new_figure(width: float = 8.0, height: float = 3.6) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _title(ax: Axes, title: str, subtitle: str) -> None:
    # Both anchored to the axes' top-left in points, so they never overlap at any height.
    ax.set_title(subtitle, loc="left", fontsize=9, color=INK_SECONDARY, pad=8)
    ax.annotate(
        title,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 26),
        textcoords="offset points",
        fontsize=13,
        fontweight="bold",
        color=INK,
        va="bottom",
    )


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _engine_bars(
    results: Mapping[str, Any],
    metric: str,
    *,
    title: str,
    subtitle: str,
    xlabel: str,
    fmt: str,
    path: Path,
    reference: float | None = None,
) -> Path:
    runs = runs_for(results, OVERALL)
    colors = engine_colors(results)
    labels = [r["label"] for r in runs]
    values = [r[metric] for r in runs]

    fig, ax = _new_figure(height=0.55 * len(runs) + 1.6)
    _style(ax, "x")
    y = range(len(runs))
    ax.barh(y, values, height=0.6, color=[colors[r["engine"]] for r in runs])
    ax.set_yticks(list(y), labels, color=INK)
    ax.invert_yaxis()  # first engine (the baseline) on top
    vmax = max(values)
    for yi, v in zip(y, values, strict=True):
        ax.text(v + vmax * 0.01, yi, fmt.format(v), va="center", fontsize=9, color=INK)
    if reference is not None:
        ax.axvline(
            reference,
            color=INK_SECONDARY,
            linestyle="--",
            linewidth=1,
            label=f"{reference:g}× = {engine_labels(results)[results['baseline']]} baseline",
        )
        ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK_SECONDARY)
    ax.set_xlim(0, vmax * 1.18)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    _title(ax, title, subtitle)
    _footnote(fig, results)
    return _save(fig, path)


def plot_throughput_mbs(results: Mapping[str, Any], path: Path) -> Path:
    return _engine_bars(
        results,
        "mb_per_s",
        title="Encoding throughput (MB/s)",
        subtitle="Full mixed corpus · higher is better",
        xlabel="Throughput (MB/s, 1 MB = 10⁶ bytes UTF-8)",
        fmt="{:,.1f} MB/s",
        path=path,
    )


def plot_throughput_mtoks(results: Mapping[str, Any], path: Path) -> Path:
    return _engine_bars(
        results,
        "mtok_per_s",
        title="Encoding throughput (Mtok/s)",
        subtitle="Full mixed corpus · higher is better",
        xlabel="Throughput (million tokens / s)",
        fmt="{:,.2f} Mtok/s",
        path=path,
    )


def plot_speedup(results: Mapping[str, Any], path: Path) -> Path:
    baseline = engine_labels(results)[results["baseline"]]
    return _engine_bars(
        results,
        "speedup_vs_baseline",
        title="Speedup vs Hugging Face Fast",
        subtitle=f"Best-time ratio relative to {baseline} · higher is better",
        xlabel="Speedup (×)",
        fmt="{:.2f}×",
        path=path,
        reference=1.0,
    )


def plot_throughput_by_domain(results: Mapping[str, Any], path: Path) -> Path:
    doms = domains(results)
    labels = engine_labels(results)
    colors = engine_colors(results)
    n = len(labels)
    group = 0.8
    width = group / n

    fig, ax = _new_figure(width=9.0, height=4.6)
    _style(ax, "y")
    vmax = 0.0
    for i, (engine, label) in enumerate(labels.items()):
        vals = [
            next(r for r in runs_for(results, d) if r["engine"] == engine)["mb_per_s"] for d in doms
        ]
        vmax = max(vmax, *vals)
        xs = [j - group / 2 + width * (i + 0.5) for j in range(len(doms))]
        bars = ax.bar(
            xs, vals, width=width, color=colors[engine], label=label, edgecolor=SURFACE, lw=1.5
        )
        ax.bar_label(bars, fmt="%.0f", fontsize=7, color=INK, padding=2)
    ax.set_xticks(range(len(doms)), doms, color=INK, fontsize=10)
    ax.set_ylim(0, vmax * 1.15)
    ax.set_ylabel("Throughput (MB/s)", color=INK_SECONDARY)
    ax.set_xlabel("Corpus domain", color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8, ncols=min(n, 4), loc="upper left", bbox_to_anchor=(0, 1.0))
    _title(ax, "Throughput by domain (MB/s)", "Each domain benchmarked separately")
    _footnote(fig, results)
    return _save(fig, path)


def write_all_plots(results: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    return {
        "mbs": plot_throughput_mbs(results, out_dir / PLOT_FILES["mbs"]),
        "mtoks": plot_throughput_mtoks(results, out_dir / PLOT_FILES["mtoks"]),
        "speedup": plot_speedup(results, out_dir / PLOT_FILES["speedup"]),
        "domain": plot_throughput_by_domain(results, out_dir / PLOT_FILES["domain"]),
    }
