import itertools
from pathlib import Path

import pytest

from poc.bench import (
    BenchmarkResult,
    Payload,
    RunStats,
    TokenCountMismatchError,
    collect_environment,
    hardware_summary,
    run_benchmark,
    time_engine,
)
from poc.corpus import read_text


class FakeEngine:
    """Counts whitespace-separated words as 'tokens'; records how often it was called."""

    def __init__(self, name: str, scale: int = 1, label: str | None = None) -> None:
        self.name = name
        self.label = label or name.upper()
        self.scale = scale
        self.calls = 0

    def encode(self, payload: Payload):
        self.calls += 1
        return [t.split() * self.scale for t in payload.texts]

    def count_tokens(self, output) -> int:
        return sum(len(row) for row in output)


class FakeTimer:
    """Deterministic clock: each start/stop pair spans ``step`` seconds, growing by 1s."""

    def __init__(self, steps):
        self._steps = iter(steps)
        self._now = 0.0
        self._started = False

    def __call__(self) -> float:
        if self._started:
            self._now += next(self._steps)
        self._started = not self._started
        return self._now


@pytest.fixture
def payload(tmp_path: Path) -> Payload:
    return Payload.from_texts("tiny", ["a b c", "ä ö\r\nü", "x"], tmp_path / "tiny.txt")


def test_payload_from_texts_writes_separated_file(payload):
    assert payload.nbytes == len(b"a b c") + len("ä ö\r\nü".encode()) + 1
    assert payload.ndocs == 3
    assert read_text(payload.path) == payload.texts


def test_time_engine_runs_warmup_and_iterations(payload):
    engine = FakeEngine("e")
    stats = time_engine(engine, payload, iterations=3, warmup=2, timer=FakeTimer([2.0, 1.0, 4.0]))
    assert engine.calls == 5
    assert stats.times_s == [2.0, 1.0, 4.0]
    assert stats.ntokens == 3 + 3 + 1
    assert stats.min_s == 1.0
    assert stats.mean_s == pytest.approx(7 / 3)
    assert stats.std_s == pytest.approx(1.527525, rel=1e-5)


def test_throughput_uses_min_time():
    s = RunStats("e", "E", "p", nbytes=4_000_000, ndocs=1, ntokens=1_000_000, times_s=[4, 2, 8])
    assert s.mb_per_s == pytest.approx(2.0)
    assert s.mtok_per_s == pytest.approx(0.5)
    d = s.to_dict()
    assert d["min_s"] == 2 and d["mb_per_s"] == pytest.approx(2.0)


def test_single_iteration_has_zero_std():
    s = RunStats("e", "E", "p", nbytes=1, ndocs=1, ntokens=1, times_s=[1.0])
    assert s.std_s == 0.0


@pytest.mark.parametrize("iterations,warmup", [(0, 1), (1, -1)])
def test_time_engine_validates_args(payload, iterations, warmup):
    with pytest.raises(ValueError):
        time_engine(FakeEngine("e"), payload, iterations=iterations, warmup=warmup)


def test_run_benchmark_speedup_and_serialisation(payload, tmp_path):
    other = Payload.from_texts("other", ["p q r s"], tmp_path / "other.txt")
    base, fast = FakeEngine("base"), FakeEngine("fast")
    # base takes 4s per pass, fast takes 1s; engines alternate per payload.
    timer = FakeTimer(itertools.cycle([4.0, 4.0, 1.0, 1.0]))
    result = run_benchmark(
        [base, fast],
        [payload, other],
        iterations=2,
        warmup=0,
        baseline="base",
        environment={"cpu_model": "Test CPU"},
        timer=timer,
    )
    assert isinstance(result, BenchmarkResult)
    assert result.engines == ["base", "fast"]
    assert result.payloads == ["tiny", "other"]
    assert result.speedup("fast", "tiny") == pytest.approx(4.0)
    assert result.speedup("base", "other") == pytest.approx(1.0)
    assert result.label("fast") == "FAST"

    d = result.to_dict()
    assert d["baseline"] == "base"
    assert d["environment"] == {"cpu_model": "Test CPU"}
    assert d["config"]["payloads"]["tiny"] == {"bytes": payload.nbytes, "documents": 3}
    assert len(d["runs"]) == 4
    assert {r["speedup_vs_baseline"] for r in d["runs"] if r["engine"] == "fast"} == {4.0}


def test_run_benchmark_detects_token_count_mismatch(payload):
    with pytest.raises(TokenCountMismatchError):
        run_benchmark(
            [FakeEngine("base"), FakeEngine("bad", scale=2)],
            [payload],
            iterations=1,
            warmup=0,
            baseline="base",
            environment={},
        )


def test_run_benchmark_requires_known_baseline(payload):
    with pytest.raises(ValueError):
        run_benchmark([FakeEngine("a")], [payload], baseline="missing", environment={})


def test_get_unknown_raises(payload):
    result = run_benchmark(
        [FakeEngine("a")], [payload], iterations=1, warmup=0, baseline="a", environment={}
    )
    with pytest.raises(KeyError):
        result.get("a", "nope")


def test_collect_environment_has_required_fields():
    env = collect_environment()
    for key in ("cpu_model", "logical_threads", "ram_gb", "os", "python", "packages"):
        assert key in env
    assert set(env["packages"]) >= {"gigatoken", "tokenizers", "transformers", "sentencepiece"}


def test_hardware_summary_formats():
    env = {"cpu_model": "CPU X", "physical_cores": 8, "logical_threads": 16, "ram_gb": 32.0}
    assert hardware_summary(env) == "CPU X (8 cores / 16 threads, 32.0 GB RAM)"
    assert hardware_summary({"cpu_model": "C", "logical_threads": 4}) == "C (4 threads)"
