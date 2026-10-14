"""Run the tokenizer throughput benchmark and write results + graphs.

Usage:
    uv run python scripts/run_benchmark.py --size-mb 50 --output-dir results/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from poc import TOKENIZER_ID
from poc.bench import Payload, RunStats, build_engines, collect_environment, run_benchmark
from poc.corpus import DOMAINS, generate_corpus
from poc.plots import write_all_plots
from poc.reporter import summary_markdown, write_json, write_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--size-mb", type=float, default=50, help="target corpus size in MB (10^6 B)")
    p.add_argument("--iterations", type=int, default=3, help="timed iterations per engine")
    p.add_argument("--warmup", type=int, default=1, help="untimed warmup passes per engine")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--output-markdown", type=Path, default=None, help="extra summary copy")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="corpus files (ignored)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", default=TOKENIZER_ID)
    p.add_argument(
        "--no-extra-baseline",
        action="store_true",
        help="skip the supplementary HF encode_batch_fast engine",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.iterations < 3:
        print("warning: fewer than 3 timed iterations; results will be noisy", file=sys.stderr)

    t0 = time.perf_counter()
    corpus = generate_corpus(int(args.size_mb * 1e6), seed=args.seed)
    by_domain = corpus.by_domain()
    print(
        f"corpus: {corpus.nbytes / 1e6:.1f} MB, {len(corpus):,} docs "
        f"({', '.join(f'{d} {b / 1e6:.1f} MB' for d, b in corpus.domain_bytes().items())}) "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    data = args.data_dir
    payloads = [Payload.from_texts("all", corpus.texts, data / "corpus_all.txt")]
    payloads += [
        Payload.from_texts(d, [doc.text for doc in by_domain[d]], data / f"corpus_{d}.txt")
        for d in DOMAINS
    ]
    corpus.write_jsonl(data / "corpus.jsonl")

    engines = build_engines(args.tokenizer, include_extra=not args.no_extra_baseline)

    def progress(s: RunStats) -> None:
        print(
            f"  {s.payload:<6} {s.label:<32} best {s.min_s:7.3f}s  "
            f"{s.mb_per_s:8.1f} MB/s  {s.mtok_per_s:7.2f} Mtok/s",
            flush=True,
        )

    result = run_benchmark(
        engines,
        payloads,
        iterations=args.iterations,
        warmup=args.warmup,
        environment=collect_environment(),
        progress=progress,
    )
    results = result.to_dict()
    results["config"].update(size_mb=args.size_mb, seed=args.seed)

    out = args.output_dir
    write_json(results, out / "benchmark_results.json")
    write_summary(results, out / "benchmark_summary.md")
    if args.output_markdown is not None:
        write_summary(results, args.output_markdown)
    plots = write_all_plots(results, out)

    print()
    print(summary_markdown(results))
    print(f"wrote {out / 'benchmark_results.json'}, {out / 'benchmark_summary.md'}")
    for path in plots.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
