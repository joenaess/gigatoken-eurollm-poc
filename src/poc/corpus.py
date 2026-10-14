"""Deterministic synthetic corpus approximating the OpenEuroLLM training mix.

The corpus is built from embedded seed vocabularies and templates (no network access, no
external datasets) so that a given ``(size, seed)`` always yields byte-identical output.

Domain mix (by UTF-8 bytes):

* ``prose`` (70%): English and European multilingual prose, including the scripts that
  received v2 fixes (Georgian, Greek, Bulgarian, Ukrainian) alongside Latin-script languages.
* ``code`` (15%): Python and structured data with deep (up to 32-space) indentation and tabs.
* ``chat`` (10%): ChatML multi-turn conversations with ``<think>`` reasoning blocks.
* ``math`` (5%): LaTeX/plain equations interleaved with FIM and StarCoder markers.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

DOC_SEPARATOR = "<|endoftext|>\n"

DOMAIN_SHARES: dict[str, float] = {
    "prose": 0.70,
    "code": 0.15,
    "chat": 0.10,
    "math": 0.05,
}
DOMAINS: tuple[str, ...] = tuple(DOMAIN_SHARES)


def _words(text: str) -> list[str]:
    return text.split()


# Seed vocabularies: common words per language. Sentences are sampled from these with
# light morphology noise (numbers, capitalisation, punctuation) to avoid trivially
# repetitive text.
WORDS: dict[str, list[str]] = {
    "en": _words(
        "the of and to in is that for it as with was on be by this are from at or which "
        "language model european research data training open source community university "
        "government policy history science people city river mountain winter summer "
        "important different public between because through however although several "
        "development knowledge network infrastructure evaluation benchmark throughput"
    ),
    "de": _words(
        "der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als "
        "auch es an werden aus er hat dass sie nach wird bei einer um am sind noch wie "
        "Sprache Forschung Gesellschaft Wissenschaft Entwicklung Universität Regierung "
        "Straßenbahn Fußgängerzone Größenordnung Übersetzung Datenverarbeitung Gemeinde"
    ),
    "fr": _words(
        "le la les de des du un une et en est que qui dans pour pas sur au avec ce il "
        "elle nous vous ils sont été être fait plus comme mais tout bien où même langue "
        "recherche société université développement données évaluation français "
        "européenne hôpital château fenêtre élève garçon cœur naïve"
    ),
    "sv": _words(
        "och att det som en på är av för med till den har de inte om ett han men var jag "
        "sig från vi så kan man när år säger hon under också efter eller språk forskning "
        "samhälle universitet utveckling regering kommun sjö räksmörgås fjäll ö ängel "
        "höst kärlek björk"
    ),
    "es": _words(
        "el la de que y a en un ser se no haber por con su para como estar tener le lo "
        "todo pero más hacer o poder decir este ir otro ese si me ya ver porque dar "
        "lengua investigación sociedad universidad desarrollo gobierno año niño corazón "
        "canción camión señal"
    ),
    "pl": _words(
        "i w nie na się z do to że jest jak co ale o po tak za od jego tylko już tym być "
        "może bardzo przez który mnie było język badania społeczeństwo uniwersytet rozwój "
        "rząd źródło żółw gęś łódź świat miłość książka"
    ),
    "fi": _words(
        "ja on ei se että hän oli ovat mutta kun niin myös vain jos sekä tai kuin kieli "
        "tutkimus yhteiskunta yliopisto kehitys hallitus järvi metsä tietokone kirjasto "
        "äiti yö sää työ yhdessä päivä kesä talvi"
    ),
    "ka": _words(
        "და არის ეს რომ არ იყო ის მე შენ ჩვენ თქვენ ისინი ერთი ორი სამი ენა კვლევა "
        "საზოგადოება უნივერსიტეტი განვითარება მთავრობა ქალაქი მდინარე მთა ზამთარი ზაფხული "
        "ისტორია მეცნიერება ხალხი წიგნი სიყვარული საქართველო თბილისი ქართული დამწერლობა "
        "მხედრული ანბანი"
    ),
    "el": _words(
        "και το η να του με την για από που στο είναι σε τα δεν θα οι ένα στην γλώσσα "
        "έρευνα κοινωνία πανεπιστήμιο ανάπτυξη κυβέρνηση πόλη ποτάμι βουνό χειμώνας "
        "καλοκαίρι ιστορία επιστήμη άνθρωποι βιβλίο αγάπη Ελλάδα Αθήνα ελληνικά αλφάβητο "
        "ψυχή ωκεανός"
    ),
    "bg": _words(
        "и в на да се за не с от че е по като са това което той тя ние те един език "
        "изследване общество университет развитие правителство град река планина зима "
        "лято история наука хора книга любов България София български азбука щастие ъгъл "
        "юг ябълка"
    ),
    "uk": _words(
        "і в на що не з до це як та у він вона ми ви вони один два три мова дослідження "
        "суспільство університет розвиток уряд місто річка гора зима літо історія наука "
        "люди книга любов Україна Київ українська абетка ґанок їжак єдність щастя"
    ),
}

# Weights within the prose domain: English dominates, the rest reflects European coverage
# with an emphasis on the scripts that received v2 fixes.
PROSE_LANG_WEIGHTS: dict[str, float] = {
    "en": 0.30,
    "de": 0.08,
    "fr": 0.08,
    "sv": 0.06,
    "es": 0.07,
    "pl": 0.06,
    "fi": 0.05,
    "ka": 0.07,
    "el": 0.08,
    "bg": 0.07,
    "uk": 0.08,
}

CODE_IDENTS = _words(
    "data value result items config node tree buffer index offset token tokenizer "
    "batch payload request response cache state path frame record vector matrix weight"
)
CODE_TYPES = ("int", "str", "float", "bool", "bytes", "list[int]", "dict[str, Any]")

CHAT_SYSTEM = [
    "You are a helpful multilingual assistant.",
    "Du bist ein hilfreicher Assistent.",
    "Είσαι ένας χρήσιμος βοηθός.",
    "Ти корисний помічник.",
    "Vous êtes un assistant utile.",
]

STARCODER_MARKERS = ["<filename>", "<file_sep>", "<reponame>", "<jupyter_code>"]
FIM_MARKERS = ["<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<fim_pad>"]
MATH_SYMBOLS = ["+", "-", "\\cdot", "\\times", "/", "=", "\\leq", "\\geq", "\\approx"]
MATH_FUNCS = ["\\sin", "\\cos", "\\log", "\\exp", "\\sqrt", "\\tanh"]
GREEK = ["\\alpha", "\\beta", "\\gamma", "\\lambda", "\\mu", "\\sigma", "\\theta", "\\omega"]


@dataclass(frozen=True, slots=True)
class Document:
    """A single corpus document tagged with its domain (and language for prose)."""

    domain: str
    text: str
    lang: str | None = None

    @property
    def nbytes(self) -> int:
        return len(self.text.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class Corpus:
    """An ordered collection of documents with per-domain helpers and exporters."""

    documents: tuple[Document, ...]
    seed: int

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    @property
    def nbytes(self) -> int:
        return sum(d.nbytes for d in self.documents)

    @property
    def texts(self) -> list[str]:
        return [d.text for d in self.documents]

    def by_domain(self) -> dict[str, list[Document]]:
        out: dict[str, list[Document]] = {d: [] for d in DOMAINS}
        for doc in self.documents:
            out[doc.domain].append(doc)
        return out

    def domain_bytes(self) -> dict[str, int]:
        return {k: sum(d.nbytes for d in v) for k, v in self.by_domain().items()}

    def write_text(self, path: Path, domain: str | None = None) -> Path:
        """Write documents joined by :data:`DOC_SEPARATOR` (no trailing separator)."""
        docs = self.documents if domain is None else self.by_domain()[domain]
        return write_text(path, (d.text for d in docs))

    def write_jsonl(self, path: Path) -> Path:
        """Write one JSON object per line with ``text``, ``domain`` and ``lang`` fields."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for doc in self.documents:
                row = {"text": doc.text, "domain": doc.domain, "lang": doc.lang}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path


def write_text(path: Path, texts: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" keeps \r\n and lone \r bytes intact; they matter for tokenization.
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(DOC_SEPARATOR.join(texts))
    return path


def read_text(path: Path) -> list[str]:
    """Inverse of :func:`write_text`: split a multi-document file on the separator."""
    with path.open(encoding="utf-8", newline="") as fh:
        return [t for t in fh.read().split(DOC_SEPARATOR) if t]


def read_jsonl(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [Document(domain=r["domain"], text=r["text"], lang=r.get("lang")) for r in rows]


# --------------------------------------------------------------------------- generators


def _sentence(rng: random.Random, lang: str) -> str:
    words = rng.choices(WORDS[lang], k=rng.randint(6, 18))
    if rng.random() < 0.25:
        words.insert(rng.randrange(len(words)), str(rng.randint(1, 2026)))
    if rng.random() < 0.15:
        words.insert(rng.randrange(len(words)), f"({rng.choice(WORDS[lang])})")
    if rng.random() < 0.2:
        i = rng.randrange(len(words))
        words[i] = words[i] + ","
    words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words) + rng.choice(".....?!:")


def gen_prose(rng: random.Random) -> Document:
    langs = list(PROSE_LANG_WEIGHTS)
    lang = rng.choices(langs, weights=list(PROSE_LANG_WEIGHTS.values()))[0]
    title = " ".join(rng.choices(WORDS[lang], k=rng.randint(2, 5))).title()
    paragraphs = [
        " ".join(_sentence(rng, lang) for _ in range(rng.randint(2, 7)))
        for _ in range(rng.randint(2, 8))
    ]
    sep = rng.choice(["\n\n", "\n\n", "\n", "\r\n\r\n"])
    return Document("prose", f"{title}{sep}{sep.join(paragraphs)}", lang=lang)


def _code_block(rng: random.Random, depth: int, indent: str, budget: list[int]) -> list[str]:
    """Recursively emit nested Python statements; ``budget`` caps total line count."""
    pad = indent * depth
    lines: list[str] = []
    for _ in range(rng.randint(1, 4)):
        if budget[0] <= 0:
            break
        budget[0] -= 1
        a, b = rng.sample(CODE_IDENTS, 2)
        kind = rng.random()
        if kind < 0.35 and depth < 8:
            header = rng.choice(
                [
                    f"if {a} is not None and len({a}) > {rng.randint(0, 64)}:",
                    f"for {a} in {b}:",
                    f"while {a} < {rng.randint(1, 1000)}:",
                    "try:",
                    f"with open({a}_path, encoding='utf-8') as {b}:",
                ]
            )
            lines.append(pad + header)
            lines.extend(_code_block(rng, depth + 1, indent, budget) or [pad + indent + "pass"])
            if header == "try:":
                lines.append(pad + "except (KeyError, ValueError) as exc:")
                lines.append(pad + indent + f"raise RuntimeError(f'bad {a}: {{exc}}') from exc")
        elif kind < 0.7:
            lines.append(pad + f"{a} = {b}.get({a!r}, {rng.randint(0, 99)}) * {rng.random():.4f}")
        elif kind < 0.85:
            lines.append(pad + f"# {_sentence(rng, 'en')}")
        else:
            lines.append(pad + f"{a}.append(({b}, {rng.randint(0, 9999)!r}))")
    return lines


def gen_code(rng: random.Random) -> Document:
    flavour = rng.random()
    if flavour < 0.75:
        indent = rng.choice(["    ", "    ", "  ", "\t"])
        out = ["from __future__ import annotations", "", "from typing import Any", ""]
        for _ in range(rng.randint(1, 3)):
            name = rng.choice(CODE_IDENTS).title()
            out.append(f"class {name}Processor:")
            out.append(f'{indent}"""{_sentence(rng, "en")}"""')
            out.append("")
            for _ in range(rng.randint(1, 3)):
                fn, arg = rng.sample(CODE_IDENTS, 2)
                ty = rng.choice(CODE_TYPES)
                out.append(f"{indent}def {fn}(self, {arg}: {ty}) -> {ty}:")
                out.extend(_code_block(rng, 2, indent, [rng.randint(8, 40)]))
                out.append(f"{indent * 2}return {arg}")
                out.append("")
        text = "\n".join(out)
    else:
        # Structured data: deeply nested JSON with wide indentation.
        def nest(d: int) -> object:
            if d == 0 or rng.random() < 0.25:
                return rng.choice([rng.randint(0, 10**6), rng.random(), rng.choice(CODE_IDENTS)])
            return {rng.choice(CODE_IDENTS): nest(d - 1) for _ in range(rng.randint(1, 4))}

        text = json.dumps(nest(8), indent=rng.choice([2, 4, 8]), ensure_ascii=False)
    return Document("code", text, lang="python" if flavour < 0.75 else "json")


def gen_chat(rng: random.Random) -> Document:
    lang = rng.choice(list(PROSE_LANG_WEIGHTS))
    parts = [f"<|im_start|>system\n{rng.choice(CHAT_SYSTEM)}<|im_end|>\n"]
    for _ in range(rng.randint(1, 4)):
        user = " ".join(_sentence(rng, lang) for _ in range(rng.randint(1, 3)))
        parts.append(f"<|im_start|>user\n{user}<|im_end|>\n")
        thought = " ".join(_sentence(rng, "en") for _ in range(rng.randint(1, 5)))
        answer = "\n\n".join(_sentence(rng, lang) for _ in range(rng.randint(1, 4)))
        parts.append(f"<|im_start|>assistant\n<think>\n{thought}\n</think>\n\n{answer}<|im_end|>\n")
    return Document("chat", "".join(parts), lang=lang)


def _expr(rng: random.Random, depth: int = 0) -> str:
    if depth > 2 or rng.random() < 0.3:
        return rng.choice([str(rng.randint(0, 999)), rng.choice(GREEK), "x", "y", "n"])
    kind = rng.random()
    if kind < 0.3:
        return f"\\frac{{{_expr(rng, depth + 1)}}}{{{_expr(rng, depth + 1)}}}"
    if kind < 0.5:
        return f"{rng.choice(MATH_FUNCS)}({_expr(rng, depth + 1)})"
    if kind < 0.6:
        return f"\\sum_{{i=1}}^{{{rng.randint(2, 100)}}} {_expr(rng, depth + 1)}_i"
    return f"{_expr(rng, depth + 1)} {rng.choice(MATH_SYMBOLS)} {_expr(rng, depth + 1)}"


def gen_math(rng: random.Random) -> Document:
    lines = []
    for _ in range(rng.randint(3, 12)):
        r = rng.random()
        if r < 0.55:
            lines.append(f"$$ {_expr(rng)} = {_expr(rng)} $$")
        elif r < 0.75:
            lines.append(f"{rng.randint(1, 10**4)} * {rng.randint(1, 10**4)} = {rng.random():.6f}")
        elif r < 0.9:
            pre, suf = _expr(rng), _expr(rng)
            lines.append(f"<fim_prefix>{pre}<fim_suffix>{suf}<fim_middle>{_expr(rng)}")
        else:
            lines.append(
                f"<reponame>eurollm/math<filename>eq_{rng.randint(0, 99)}.tex<file_sep>"
                f"{_expr(rng)}<jupyter_code>print({rng.randint(0, 9)})"
            )
    return Document("math", "\n".join(lines), lang=None)


GENERATORS = {"prose": gen_prose, "code": gen_code, "chat": gen_chat, "math": gen_math}


def generate_corpus(target_bytes: int, seed: int = 0) -> Corpus:
    """Generate a corpus of roughly ``target_bytes`` UTF-8 bytes following
    :data:`DOMAIN_SHARES`.

    Each domain draws from its own RNG (derived from ``seed``) so that changing one domain's
    generator does not perturb the others. Documents are interleaved deterministically.
    """
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    per_domain: dict[str, list[Document]] = {}
    for i, (domain, share) in enumerate(DOMAIN_SHARES.items()):
        rng = random.Random(seed * 1_000_003 + i)
        budget = max(1, round(target_bytes * share))
        docs: list[Document] = []
        total = 0
        while total < budget:
            doc = GENERATORS[domain](rng)
            docs.append(doc)
            total += doc.nbytes
        per_domain[domain] = docs

    # Deterministic shuffle so the combined file mixes domains like a real shard.
    mixed = [d for docs in per_domain.values() for d in docs]
    random.Random(seed).shuffle(mixed)
    return Corpus(documents=tuple(mixed), seed=seed)
