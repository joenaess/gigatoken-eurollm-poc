"""Derive token inventories (special / user-defined / whitespace) from the tokenizer itself.

The v2 release notes quote ~45 whitespace tokens and ~204 user-defined/special tokens; these
helpers recount them from ``tokenizer.json`` (``added_tokens``) and ``tokenizer.model``
(SentencePiece piece types) so the parity suite and reports never rely on hardcoded lists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from poc import TOKENIZER_ID

METASPACE = "▁"
WHITESPACE_CHARS = frozenset(" \t\n\r▁")
RELEASE_NOTES = {"whitespace_tokens": 45, "special_tokens": 204}


@dataclass(frozen=True)
class AddedToken:
    id: int
    content: str
    special: bool
    sp_type: str | None  # SentencePiece piece type: USER_DEFINED, CONTROL, UNKNOWN, ...


@dataclass(frozen=True)
class VocabInventory:
    added_tokens: tuple[AddedToken, ...]
    whitespace_added: tuple[AddedToken, ...]
    whitespace_pieces: tuple[tuple[int, str], ...]  # whitespace-only NORMAL pieces (id, piece)
    sp_type_counts: dict[str, int]
    vocab_size: int

    @property
    def special_count(self) -> int:
        return sum(t.special for t in self.added_tokens)

    @property
    def control_tokens(self) -> tuple[AddedToken, ...]:
        """Added tokens SentencePiece will not match in raw text (CONTROL / UNKNOWN)."""
        return tuple(t for t in self.added_tokens if t.sp_type in {"CONTROL", "UNKNOWN"})

    @property
    def whitespace_total(self) -> int:
        return len(self.whitespace_added) + len(self.whitespace_pieces)

    @property
    def literal_space_pieces(self) -> tuple[tuple[int, str], ...]:
        """Whitespace pieces spelled with a literal U+0020 instead of ``▁``.

        Spaces are escaped to ``▁`` before BPE (Metaspace / SentencePiece whitespace
        escaping, no normalizer), so these pieces cannot be produced by encoding text.
        """
        return tuple((i, p) for i, p in self.whitespace_pieces if " " in p)

    def summary(self) -> dict[str, int]:
        return {
            "vocab_size": self.vocab_size,
            "added_tokens": len(self.added_tokens),
            "added_tokens_special": self.special_count,
            "added_tokens_non_special": len(self.added_tokens) - self.special_count,
            "sp_user_defined": self.sp_type_counts.get("USER_DEFINED", 0),
            "sp_control": self.sp_type_counts.get("CONTROL", 0),
            "sp_byte": self.sp_type_counts.get("BYTE", 0),
            "whitespace_added_tokens": len(self.whitespace_added),
            "whitespace_vocab_pieces": len(self.whitespace_pieces),
            "whitespace_total": self.whitespace_total,
            "whitespace_literal_space_pieces": len(self.literal_space_pieces),
            "whitespace_reachable": self.whitespace_total - len(self.literal_space_pieces),
        }


def is_whitespace(text: str) -> bool:
    return bool(text) and all(c in WHITESPACE_CHARS for c in text)


def piece_to_text(piece: str) -> str:
    """Map a vocab piece to the raw text that produces it (``▁`` stands for a space)."""
    return piece.replace(METASPACE, " ")


def tokenizer_files(tokenizer_id: str = TOKENIZER_ID) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    return (
        Path(hf_hub_download(tokenizer_id, "tokenizer.json")),
        Path(hf_hub_download(tokenizer_id, "tokenizer.model")),
    )


@cache
def load_inventory(tokenizer_id: str = TOKENIZER_ID) -> VocabInventory:
    from sentencepiece import sentencepiece_model_pb2 as pb

    json_path, model_path = tokenizer_files(tokenizer_id)
    config = json.loads(json_path.read_text(encoding="utf-8"))
    model = pb.ModelProto()
    model.ParseFromString(model_path.read_bytes())
    type_name = pb.ModelProto.SentencePiece.Type.Name
    sp_types = {i: type_name(p.type) for i, p in enumerate(model.pieces)}

    type_counts: dict[str, int] = {}
    for t in sp_types.values():
        type_counts[t] = type_counts.get(t, 0) + 1

    added = tuple(
        AddedToken(
            id=int(t["id"]),
            content=str(t["content"]),
            special=bool(t.get("special")),
            sp_type=sp_types.get(int(t["id"])),
        )
        for t in config.get("added_tokens", [])
    )
    added_ids = {t.id for t in added}
    ws_pieces = tuple(
        (i, p.piece)
        for i, p in enumerate(model.pieces)
        if sp_types[i] == "NORMAL" and i not in added_ids and is_whitespace(p.piece)
    )
    return VocabInventory(
        added_tokens=added,
        whitespace_added=tuple(t for t in added if is_whitespace(t.content)),
        whitespace_pieces=ws_pieces,
        sp_type_counts=type_counts,
        vocab_size=len(model.pieces),
    )
