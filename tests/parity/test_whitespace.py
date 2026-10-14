"""Whitespace & indentation token parity.

Cases come from two sources: hand-written indentation/newline scenarios from the spec, and
every whitespace-only token found in the tokenizer's own vocabulary (added tokens such as
``\\t\\t`` or ``\\r\\n`` plus whitespace-only BPE pieces such as ``▁▁▁▁``).
"""

import pytest

from poc.vocab import load_inventory, piece_to_text

pytestmark = pytest.mark.parity

INDENTS = [2, 3, 4, 8, 16, 32]

INDENT_CASES = {
    **{f"spaces_{n}": " " * n for n in INDENTS},
    **{f"indented_stmt_{n}": f"def f():\n{' ' * n}return x\n" for n in INDENTS},
    **{f"indent_after_newline_{n}": f"\n{' ' * n}" for n in INDENTS},
    "nested_4_to_32": "".join(f"{' ' * n}x = {n}\n" for n in range(4, 36, 4)),
    "dedent_ladder": "".join(f"{' ' * n}pass\n" for n in (32, 16, 8, 4, 2, 0)),
    "tabs_1_to_9": "".join("\t" * n + "y\n" for n in range(1, 10)),
    "mixed_tab_space": "\t    \t  x\n  \t\ty",
    "spaces_over_max_piece": " " * 33 + "z" + " " * 64,
}

NEWLINE_CASES = {
    "tab_newline": "\t\n",
    "crlf": "\r\n",
    "crlf_runs": "a\r\n\r\nb\r\n\r\n\r\nc",
    "lone_cr": "a\rb\r\rc",
    "nl_2": "\n\n",
    "nl_3": "\n\n\n",
    "nl_4": "\n\n\n\n",
    "nl_7": "\n" * 7,
    "trailing_spaces_then_nl": "x   \n    \n",
    "tab_nl_tab": "\t\n\t\n\t",
    "space_cr": " \r",
    "nl_then_indent": "a\n\n\n        b",
    "only_space": " ",
    "nbsp_and_ideographic": "a b　c",
}

_inv = load_inventory()
VOCAB_WS = sorted(
    {t.content for t in _inv.whitespace_added}
    | {piece_to_text(piece) for _, piece in _inv.whitespace_pieces}
)


@pytest.mark.parametrize("text", INDENT_CASES.values(), ids=INDENT_CASES.keys())
def test_indentation(assert_parity, text):
    assert_parity(text)


@pytest.mark.parametrize("text", NEWLINE_CASES.values(), ids=NEWLINE_CASES.keys())
def test_newline_combinations(assert_parity, text):
    assert_parity(text)


@pytest.mark.parametrize("ws", VOCAB_WS, ids=[repr(w) for w in VOCAB_WS])
def test_vocab_whitespace_token_alone(assert_parity, ws):
    assert_parity(ws)


@pytest.mark.parametrize("ws", VOCAB_WS, ids=[repr(w) for w in VOCAB_WS])
def test_vocab_whitespace_token_in_context(assert_parity, ws):
    assert_parity(f"if x:{ws}return{ws}y")


@pytest.mark.parametrize(
    "token", _inv.whitespace_added, ids=[repr(t.content) for t in _inv.whitespace_added]
)
def test_whitespace_added_token_maps_to_single_id(encoders, token):
    """User-defined whitespace tokens must encode to exactly their own ID on every path."""
    results = encoders.encode_all(token.content)
    assert all(ids == [token.id] for ids in results.values()), results


def test_whitespace_inventory_is_derived(inventory):
    # Sanity: the derivation found the indentation pieces the spec cares about.
    pieces = {piece_to_text(p) for _, p in inventory.whitespace_pieces}
    assert {" " * 2, " " * 4, " " * 8, " " * 16} <= pieces
    assert {"\t", "\r\n", "\t\n", "\n\n\n"} <= {t.content for t in inventory.whitespace_added}


_LITERAL = _inv.literal_space_pieces


@pytest.mark.parametrize("piece_id,piece", _LITERAL, ids=[repr(p) for _, p in _LITERAL])
def test_literal_space_pieces_are_never_emitted(encoders, piece_id, piece):
    """Pieces spelled with U+0020 are unreachable: spaces become ``▁`` before BPE."""
    for text in (piece, f"x{piece}y", f"def f():\n{piece}return 1\n"):
        results = encoders.encode_all(text)
        assert len({tuple(ids) for ids in results.values()}) == 1, results
        assert piece_id not in results["hf_raw"]
