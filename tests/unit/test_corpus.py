import json

import pytest

from poc.corpus import (
    DOC_SEPARATOR,
    DOMAIN_SHARES,
    DOMAINS,
    Corpus,
    generate_corpus,
    read_jsonl,
    read_text,
)

SIZE = 1_000_000


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return generate_corpus(SIZE, seed=7)


def test_deterministic_for_same_seed(corpus):
    again = generate_corpus(SIZE, seed=7)
    assert again.documents == corpus.documents


def test_different_seed_differs(corpus):
    assert generate_corpus(SIZE, seed=8).documents != corpus.documents


def test_reaches_target_size(corpus):
    assert SIZE <= corpus.nbytes < SIZE * 1.1


def test_domain_shares_are_balanced(corpus):
    per_domain = corpus.domain_bytes()
    assert set(per_domain) == set(DOMAINS)
    for domain, share in DOMAIN_SHARES.items():
        actual = per_domain[domain] / corpus.nbytes
        assert actual == pytest.approx(share, abs=0.02), domain


def test_every_document_is_tagged_and_nonempty(corpus):
    for doc in corpus:
        assert doc.domain in DOMAINS
        assert doc.text
        assert DOC_SEPARATOR not in doc.text


def test_domains_are_interleaved(corpus):
    first = [d.domain for d in corpus.documents[:50]]
    assert len(set(first)) > 1


def test_prose_covers_v2_scripts(corpus):
    langs = {d.lang for d in corpus.by_domain()["prose"]}
    assert {"en", "ka", "el", "bg", "uk"} <= langs


def test_code_has_deep_indentation():
    corpus = generate_corpus(400_000, seed=1)
    code = "\n".join(d.text for d in corpus.by_domain()["code"])
    assert "\n" + " " * 16 in code
    assert "\t\t" in code


def test_chat_uses_chatml_and_think(corpus):
    for doc in corpus.by_domain()["chat"]:
        assert doc.text.startswith("<|im_start|>system\n")
        assert "<think>" in doc.text and "</think>" in doc.text
        assert doc.text.count("<|im_start|>") == doc.text.count("<|im_end|>")


def test_math_contains_markers(corpus):
    math = "\n".join(d.text for d in corpus.by_domain()["math"])
    assert "<fim_prefix>" in math
    assert "<file_sep>" in math
    assert "\\frac" in math


def test_text_roundtrip(tmp_path, corpus):
    path = corpus.write_text(tmp_path / "corpus.txt")
    assert read_text(path) == corpus.texts
    assert not path.read_bytes().endswith(DOC_SEPARATOR.encode())


def test_text_per_domain(tmp_path, corpus):
    path = corpus.write_text(tmp_path / "code.txt", domain="code")
    assert read_text(path) == [d.text for d in corpus.by_domain()["code"]]


def test_jsonl_roundtrip(tmp_path, corpus):
    path = corpus.write_jsonl(tmp_path / "corpus.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(corpus)
    assert set(json.loads(lines[0])) == {"text", "domain", "lang"}
    assert tuple(read_jsonl(path)) == corpus.documents


def test_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        generate_corpus(0)
