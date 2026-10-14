"""Multilingual script parity, focused on scripts with v2 updates or byte-fallback fixes.

Includes natural sentences, rare/archaic code points that force byte fallback, Unicode
normalisation variants (the tokenizer has no normalizer, so NFC and NFD must stay distinct),
and whole documents from the synthetic benchmark corpus.
"""

import unicodedata

import pytest

from poc.corpus import DOMAINS, generate_corpus

pytestmark = pytest.mark.parity

SENTENCES = {
    "ka": [
        "საქართველო მდებარეობს კავკასიაში, შავი ზღვის აღმოსავლეთ სანაპიროზე.",
        "ქართული დამწერლობის სამი სახეობაა: ასომთავრული, ნუსხური და მხედრული.",
        "ⴀⴁⴂ ႠႡႢ — ნუსხური და ასომთავრული ასოები.",
        "Ა Ბ Გ — მთავრული (Mtavruli) ასოები Unicode 11-დან.",
    ],
    "el": [
        "Η Ελλάδα είναι χώρα της νοτιοανατολικής Ευρώπης.",
        "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν.",
        "ΑΒΓΔΕΖΗΘ αβγδεζηθ ς σ ϐ ϑ ϕ ϖ ϰ ϱ ϲ",
        "Το ερωτηματικό είναι «;» και όχι «?»;",
    ],
    "bg": [
        "България е държава в Югоизточна Европа.",
        "Щъркелът отлетя на юг през есента, а ъгълът остана празен.",
        "Ѫ ѫ Ѣ ѣ — старобългарски букви (юс голям, ят).",
    ],
    "uk": [
        "Україна — держава у Східній Європі.",
        "Ґанок, їжак, єнот і йогурт: ґ є ї й.",
        "Апостроф: м'ята, п’ять, сім'я, об’єкт.",
    ],
    "latin": [
        "The quick brown fox jumps over the lazy dog.",
        "Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich.",
        "Ça m'a coûté « cher », dit l'élève naïve à l'hôpital.",
        "Pchnąć w tę łódź jeża lub ośm skrzyń fig.",
        "Ærøskøbing, Smørrebrød, Þórsmörk, Øresund, Ğüşçöı.",
        "Hyvää päivää! Räksmörgås på Åland.",
    ],
}

BYTE_FALLBACK = {
    "emoji_zwj": "👩‍👩‍👧‍👦 🏳️‍🌈 🤖",
    "cjk_ext_b": "𠀀𠀁𠀂 𪚥",
    "math_alnum": "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝒜𝓁𝓅𝒽𝒶",
    "glagolitic": "ⰀⰁⰂⰃ ⰰⰱⰲⰳ",
    "combining_stack": "Z̴̢̛a̷͎͝l̸̰̈g̶͈̓o̵̗͒",
    "control_chars": "a\u0000b\u0007c\u001bd\u007f",
    "private_use": "",
    "replacement_char": "bad � byte",
    "rtl_mixed": "Hello שלום مرحبا world",
    "bom_and_zwsp": "﻿a​b‍c",
}

_SENTENCE_CASES = [(lang, s) for lang, ss in SENTENCES.items() for s in ss]


@pytest.mark.parametrize(
    "lang,text", _SENTENCE_CASES, ids=[f"{lang}-{i}" for i, (lang, _) in enumerate(_SENTENCE_CASES)]
)
def test_sentences(assert_parity, lang, text):
    assert_parity(text)


@pytest.mark.parametrize("lang", SENTENCES)
def test_paragraph_per_script(assert_parity, lang):
    assert_parity("\n\n".join(SENTENCES[lang]))


@pytest.mark.parametrize("text", BYTE_FALLBACK.values(), ids=BYTE_FALLBACK.keys())
def test_byte_fallback(assert_parity, text):
    assert_parity(text)


@pytest.mark.parametrize("form", ["NFC", "NFD", "NFKC", "NFKD"])
@pytest.mark.parametrize("lang", ["el", "uk", "latin"])
def test_unicode_normalisation_forms(assert_parity, lang, form):
    assert_parity(unicodedata.normalize(form, " ".join(SENTENCES[lang])))


def test_no_normalizer_keeps_nfc_nfd_distinct(assert_parity):
    nfc = unicodedata.normalize("NFC", "é")
    nfd = unicodedata.normalize("NFD", "é")
    assert assert_parity(nfc) != assert_parity(nfd)


@pytest.fixture(scope="module")
def corpus_by_domain():
    return {
        k: [d.text for d in v] for k, v in generate_corpus(400_000, seed=123).by_domain().items()
    }


@pytest.mark.parametrize("domain", DOMAINS)
def test_corpus_documents_batch_parity(encoders, corpus_by_domain, domain):
    """Whole synthetic documents, encoded with each path's batch API."""
    texts = corpus_by_domain[domain]
    results = {name: [list(r) for r in fn(texts)] for name, fn in encoders.batch_funcs.items()}
    reference = results["hf_raw"]
    for name, rows in results.items():
        bad = [i for i, (a, b) in enumerate(zip(reference, rows, strict=True)) if a != b]
        assert not bad, f"{name} differs from hf_raw on {len(bad)} {domain} docs, first: {bad[:5]}"
