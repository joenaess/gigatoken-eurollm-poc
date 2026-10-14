"""Shared fixtures and the parity-report hook.

Real tokenizers are only loaded when a test requests them, so ``pytest tests/unit`` never
touches the network. Parity tests download ``openeurollm/tokenizer-256k-v2`` on first use.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from poc import TOKENIZER_ID

ENCODER_NAMES = ("hf_auto", "hf_raw", "sp", "gt_compat", "gt_native")
REPO_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_TEXTS = {
    "en": "The quick brown fox jumps over the lazy dog.",
    "de": "Größenordnung der Straßenbahn-Übersetzung.",
    "ka": "ქართული ენა ერთ-ერთი უძველესი ენაა.",
    "el": "Η γλώσσα είναι ζωντανός οργανισμός.",
    "bg": "Българският език използва кирилица.",
    "uk": "Українська мова має ґ, є, ї та і.",
    "code": "def f(x):\n    if x:\n        return x\n",
}


class Encoders:
    """Every encoding path under test, normalised to ``text -> list[int]`` without BOS."""

    def __init__(self, tokenizer_id: str = TOKENIZER_ID) -> None:
        import gigatoken as gt
        import sentencepiece as spm
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer

        from poc.vocab import tokenizer_files

        json_path, model_path = tokenizer_files(tokenizer_id)
        hf_auto = AutoTokenizer.from_pretrained(tokenizer_id)
        hf_raw = Tokenizer.from_file(str(json_path))
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        gt_compat = gt.Tokenizer(hf_auto).as_hf()
        gt_native = gt.Tokenizer(tokenizer_id)

        self.funcs: dict[str, Callable[[str], list[int]]] = {
            "hf_auto": lambda t: list(hf_auto(t, add_special_tokens=False)["input_ids"]),
            "hf_raw": lambda t: hf_raw.encode(t, add_special_tokens=False).ids,
            "sp": lambda t: list(sp.encode(t)),
            # Batch API on purpose: it is the path the benchmark measures.
            "gt_compat": lambda t: list(gt_compat([t], add_special_tokens=False)["input_ids"][0]),
            "gt_native": lambda t: gt_native.encode(t).tolist(),
        }
        self.batch_funcs: dict[str, Callable[[list[str]], list[list[int]]]] = {
            "hf_raw": lambda ts: [e.ids for e in hf_raw.encode_batch(ts, add_special_tokens=False)],
            "sp": lambda ts: [list(r) for r in sp.encode(ts)],
            "gt_compat": lambda ts: gt_compat(ts, add_special_tokens=False)["input_ids"],
            "gt_native": lambda ts: gt_native.encode_batch_list(ts),
        }
        self.decode = lambda ids: hf_raw.decode(ids, skip_special_tokens=False)

    def encode_all(self, text: str, names: tuple[str, ...] = ENCODER_NAMES) -> dict[str, list]:
        return {name: self.funcs[name](text) for name in names}


@pytest.fixture(scope="session")
def encoders() -> Encoders:
    return Encoders()


@pytest.fixture(scope="session")
def inventory():
    from poc.vocab import load_inventory

    return load_inventory()


@pytest.fixture(scope="session")
def sample_texts() -> dict[str, str]:
    return dict(SAMPLE_TEXTS)


def _format_mismatch(text: str, results: dict[str, list]) -> str:
    groups: dict[str, list[str]] = defaultdict(list)
    for name, ids in results.items():
        groups[repr(ids)].append(name)
    lines = [f"token-ID mismatch for {text!r}:"]
    lines += [f"  {', '.join(names)}: {ids}" for ids, names in groups.items()]
    return "\n".join(lines)


@pytest.fixture(scope="session")
def assert_parity(encoders: Encoders) -> Callable[..., list[int]]:
    """Encode ``text`` with every path and assert the IDs are identical; returns the IDs."""

    def check(text: str, names: tuple[str, ...] = ENCODER_NAMES) -> list[int]:
        results = encoders.encode_all(text, names)
        reference = results[names[0]]
        if any(ids != reference for ids in results.values()):
            pytest.fail(_format_mismatch(text, results), pytrace=False)
        return reference

    return check


# --------------------------------------------------------------------------- parity report


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--parity-report",
        default=str(REPO_ROOT / "results" / "parity_report.md"),
        help="Where to write the parity summary when parity tests run ('' disables).",
    )


_outcomes: dict[str, Counter] = defaultdict(Counter)


def _category(nodeid: str) -> str:
    return Path(nodeid.split("::", 1)[0]).stem.removeprefix("test_")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if "parity" not in report.keywords:
        return
    cat = _category(report.nodeid)
    if hasattr(report, "wasxfail"):
        if report.skipped:
            _outcomes[cat]["xfailed"] += 1
        elif report.passed and report.when == "call":
            _outcomes[cat]["xpassed"] += 1
        return
    if report.failed:
        _outcomes[cat]["failed"] += 1
    elif report.skipped:
        _outcomes[cat]["skipped"] += 1
    elif report.passed and report.when == "call":
        _outcomes[cat]["passed"] += 1


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    path = session.config.getoption("--parity-report")
    if not path or not _outcomes:
        return
    from poc.reporter import write_parity_report

    try:
        from poc.vocab import load_inventory

        inventory = load_inventory()
    except Exception:  # report still useful without the inventory section
        inventory = None
    write_parity_report(
        Path(path),
        {cat: dict(c) for cat, c in sorted(_outcomes.items())},
        inventory=inventory,
        encoders=ENCODER_NAMES,
        command="pytest " + " ".join(sys.argv[1:]),
        timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )
