"""Benchmark harness: Hugging Face tokenizers vs gigatoken (HF-compat and native paths).

Timing model
------------
For every (engine, payload) pair the harness runs ``warmup`` untimed passes followed by
``iterations`` timed passes. Each timed pass measures only the encode call with
:func:`time.perf_counter`; Python's cyclic GC is collected beforehand and disabled during the
call. Token counting and result disposal happen outside the timed region, so engines whose
outputs are Python objects (HF ``Encoding``) are not charged for the harness's bookkeeping.

The reported elapsed time is the minimum over the timed passes (the least noisy estimator of
achievable throughput); mean and standard deviation are recorded alongside it.
"""

from __future__ import annotations

import gc
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from poc import TOKENIZER_ID
from poc.corpus import DOC_SEPARATOR

BASELINE = "hf_fast"


class TokenCountMismatchError(RuntimeError):
    """Raised when engines disagree on the number of tokens for the same payload."""


@dataclass(frozen=True)
class Payload:
    """A benchmark input: the same documents in memory and on disk."""

    name: str
    texts: list[str] = field(repr=False)
    path: Path
    nbytes: int

    @property
    def ndocs(self) -> int:
        return len(self.texts)

    @classmethod
    def from_texts(cls, name: str, texts: list[str], path: Path) -> Payload:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(DOC_SEPARATOR.join(texts))
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        return cls(name=name, texts=texts, path=path, nbytes=nbytes)


class Engine(Protocol):
    """A tokenization path under test.

    ``encode`` is the timed call; ``count_tokens`` runs afterwards on its output.
    """

    name: str
    label: str

    def encode(self, payload: Payload) -> Any: ...

    def count_tokens(self, output: Any) -> int: ...


@dataclass
class RunStats:
    """Timing and size statistics for one engine on one payload."""

    engine: str
    label: str
    payload: str
    nbytes: int
    ndocs: int
    ntokens: int
    times_s: list[float]

    @property
    def min_s(self) -> float:
        return min(self.times_s)

    @property
    def mean_s(self) -> float:
        return statistics.fmean(self.times_s)

    @property
    def std_s(self) -> float:
        return statistics.stdev(self.times_s) if len(self.times_s) > 1 else 0.0

    @property
    def mb_per_s(self) -> float:
        """Throughput in MB/s (1 MB = 10**6 bytes of UTF-8 text) at the best time."""
        return self.nbytes / 1e6 / self.min_s

    @property
    def mtok_per_s(self) -> float:
        return self.ntokens / 1e6 / self.min_s

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            min_s=self.min_s,
            mean_s=self.mean_s,
            std_s=self.std_s,
            mb_per_s=self.mb_per_s,
            mtok_per_s=self.mtok_per_s,
        )
        return d


@dataclass
class BenchmarkResult:
    stats: list[RunStats]
    baseline: str
    config: dict[str, Any]
    environment: dict[str, Any]

    def get(self, engine: str, payload: str) -> RunStats:
        for s in self.stats:
            if s.engine == engine and s.payload == payload:
                return s
        raise KeyError((engine, payload))

    @property
    def engines(self) -> list[str]:
        return list(dict.fromkeys(s.engine for s in self.stats))

    @property
    def payloads(self) -> list[str]:
        return list(dict.fromkeys(s.payload for s in self.stats))

    def label(self, engine: str) -> str:
        return next(s.label for s in self.stats if s.engine == engine)

    def speedup(self, engine: str, payload: str) -> float:
        """Best-time speedup of ``engine`` over the baseline on ``payload``."""
        return self.get(self.baseline, payload).min_s / self.get(engine, payload).min_s

    def to_dict(self) -> dict[str, Any]:
        runs = []
        for s in self.stats:
            d = s.to_dict()
            d["speedup_vs_baseline"] = self.speedup(s.engine, s.payload)
            runs.append(d)
        return {
            "tokenizer": TOKENIZER_ID,
            "baseline": self.baseline,
            "config": self.config,
            "environment": self.environment,
            "runs": runs,
        }


def time_engine(
    engine: Engine,
    payload: Payload,
    *,
    iterations: int = 3,
    warmup: int = 1,
    timer: Callable[[], float] = time.perf_counter,
) -> RunStats:
    """Warm up, then time ``iterations`` encode passes of ``engine`` over ``payload``."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    for _ in range(warmup):
        engine.encode(payload)  # result discarded immediately

    times: list[float] = []
    counts: set[int] = set()
    for _ in range(iterations):
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            start = timer()
            output = engine.encode(payload)
            elapsed = timer() - start
        finally:
            if gc_was_enabled:
                gc.enable()
        times.append(elapsed)
        counts.add(engine.count_tokens(output))
        del output

    if len(counts) != 1:
        raise TokenCountMismatchError(f"{engine.name} is nondeterministic on {payload.name}")
    return RunStats(
        engine=engine.name,
        label=engine.label,
        payload=payload.name,
        nbytes=payload.nbytes,
        ndocs=payload.ndocs,
        ntokens=counts.pop(),
        times_s=times,
    )


def run_benchmark(
    engines: Sequence[Engine],
    payloads: Iterable[Payload],
    *,
    iterations: int = 3,
    warmup: int = 1,
    baseline: str = BASELINE,
    environment: dict[str, Any] | None = None,
    progress: Callable[[RunStats], None] | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    """Benchmark every engine on every payload and cross-check token counts."""
    if baseline not in {e.name for e in engines}:
        raise ValueError(f"baseline engine {baseline!r} not among engines")
    stats: list[RunStats] = []
    payloads = list(payloads)
    for payload in payloads:
        per_payload = []
        for engine in engines:
            s = time_engine(engine, payload, iterations=iterations, warmup=warmup, timer=timer)
            per_payload.append(s)
            if progress is not None:
                progress(s)
        counts = {s.engine: s.ntokens for s in per_payload}
        if len(set(counts.values())) != 1:
            raise TokenCountMismatchError(f"token counts differ on {payload.name}: {counts}")
        stats.extend(per_payload)
    config = {
        "iterations": iterations,
        "warmup": warmup,
        "payloads": {p.name: {"bytes": p.nbytes, "documents": p.ndocs} for p in payloads},
        "timer": "time.perf_counter, min over iterations",
    }
    return BenchmarkResult(
        stats=stats,
        baseline=baseline,
        config=config,
        environment=environment if environment is not None else collect_environment(),
    )


# --------------------------------------------------------------------------- engines


class HFFastEngine:
    """Baseline: ``tokenizers.Tokenizer.encode_batch`` (the Rust "Fast" tokenizer)."""

    name = "hf_fast"
    label = "HF Fast (encode_batch)"

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def encode(self, payload: Payload) -> Any:
        return self.tokenizer.encode_batch(payload.texts, add_special_tokens=False)

    def count_tokens(self, output: Any) -> int:
        return sum(len(e.ids) for e in output)


class HFFastNoOffsetsEngine(HFFastEngine):
    """Supplementary baseline: ``encode_batch_fast`` skips offset tracking."""

    name = "hf_fast_no_offsets"
    label = "HF Fast (encode_batch_fast)"

    def encode(self, payload: Payload) -> Any:
        return self.tokenizer.encode_batch_fast(payload.texts, add_special_tokens=False)


class GTCompatEngine:
    """gigatoken behind the transformers API: ``gt.Tokenizer(hf).as_hf()(texts)``.

    ``HFCompat`` has no ``encode_batch`` (see README); batching goes through ``__call__``.
    """

    name = "gt_compat"
    label = "Gigatoken HF compat"

    def __init__(self, compat: Any) -> None:
        self.compat = compat

    def encode(self, payload: Payload) -> Any:
        return self.compat(payload.texts, add_special_tokens=False, return_attention_mask=False)

    def count_tokens(self, output: Any) -> int:
        return sum(len(row) for row in output["input_ids"])


class GTNativeEngine:
    """gigatoken native file path: ``encode_files(TextFileSource(..., separator))``.

    Unlike the in-memory engines, this includes reading and splitting the file (served from
    the page cache after warmup), mirroring how it is used on real shards.
    """

    name = "gt_native"
    label = "Gigatoken native (encode_files)"

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def encode(self, payload: Payload) -> Any:
        import gigatoken as gt

        source = gt.TextFileSource([str(payload.path)], separator=DOC_SEPARATOR)
        return self.tokenizer.encode_files(source)

    def count_tokens(self, output: Any) -> int:
        import awkward as ak

        return int(ak.sum(ak.num(output, axis=1)))


def build_engines(tokenizer_id: str = TOKENIZER_ID, *, include_extra: bool = True) -> list[Any]:
    """Instantiate all engines against the same tokenizer (downloads it if needed)."""
    import gigatoken as gt
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    raw = Tokenizer.from_file(hf_hub_download(tokenizer_id, "tokenizer.json"))
    hf = AutoTokenizer.from_pretrained(tokenizer_id)
    engines: list[Any] = [HFFastEngine(raw)]
    if include_extra:
        engines.append(HFFastNoOffsetsEngine(raw))
    engines += [
        GTCompatEngine(gt.Tokenizer(hf).as_hf()),
        GTNativeEngine(gt.Tokenizer(tokenizer_id)),
    ]
    return engines


# --------------------------------------------------------------------------- environment


def _read(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _cpu_model() -> str:
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _physical_cores() -> int | None:
    cores = set()
    phys = core = None
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("physical id"):
            phys = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core = line.split(":", 1)[1].strip()
            cores.add((phys, core))
    return len(cores) or None


def _ram_gb() -> float | None:
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 1024**2, 1)
    return None


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    return {
        "cpu_model": _cpu_model(),
        "physical_cores": _physical_cores(),
        "logical_threads": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "packages": {
            p: _version(p)
            for p in ("gigatoken", "tokenizers", "transformers", "sentencepiece", "numpy")
        },
        "thread_env": {
            k: os.environ.get(k) for k in ("RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM")
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def hardware_summary(env: dict[str, Any]) -> str:
    """One-line hardware description used in plot footnotes and tables."""
    cores = env.get("physical_cores")
    threads = env.get("logical_threads")
    core_txt = f"{cores} cores / {threads} threads" if cores else f"{threads} threads"
    ram = f", {env['ram_gb']} GB RAM" if env.get("ram_gb") else ""
    return f"{env.get('cpu_model', 'unknown CPU')} ({core_txt}{ram})"
