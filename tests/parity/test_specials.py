"""User-defined & special token parity: ChatML, reasoning, FIM and StarCoder markers.

Every entry of ``added_tokens`` is tested alone and in context. Raw SentencePiece does not
match CONTROL/UNKNOWN pieces (``<unk>``, ``<bos>``, ``<eos>``, ``<pad>``) in plain text by
design, so for those four the SentencePiece comparison is a strict xfail while the remaining
paths must still agree.
"""

import pytest

from poc.vocab import load_inventory

pytestmark = pytest.mark.parity

_inv = load_inventory()
NON_SP = ("hf_auto", "hf_raw", "gt_compat", "gt_native")
CONTROL_IDS = {t.id for t in _inv.control_tokens}

SPEC_MARKERS = {
    "starcoder": ["<filename>", "<file_sep>", "<reponame>", "<jupyter_code>"],
    "fim": ["<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<fim_pad>"],
    "chat_reasoning": ["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
}

SCENARIOS = {
    "chatml_multiturn": (
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nSimple arithmetic.\n</think>\n\n4<|im_end|>\n"
    ),
    "think_empty": "<think></think>",
    "think_nested_ws": "<think>\n\n\n    step 1\n\t\tstep 2\r\n</think>",
    "fim_psm": "<fim_prefix>def add(a, b):\n    <fim_suffix>\n    return c<fim_middle>c = a + b",
    "fim_pad_run": "<fim_pad>" * 8,
    "starcoder_repo": (
        "<reponame>eurollm/poc<filename>src/main.py\nimport os\n<file_sep>"
        "<filename>README.md\n# Title\n"
    ),
    "jupyter": "<jupyter_start><jupyter_code>print(1)<jupyter_output>1<jupyter_text>done",
    "markers_glued": "a<|im_end|><|im_start|>b<think>c</think>d",
    "partial_markers": "<|im_start <think <fim_prefix|> </think",
    "marker_with_whitespace_tokens": "\t<fim_pad>\n\n\n<file_sep>\r\n<think>\t\n",
    "tool_call": '<tool_call>{"name": "f", "args": {}}</tool_call>',
}


def _case_ids(tokens):
    return [f"{t.id}:{t.content!r}" for t in tokens]


@pytest.mark.parametrize("category,token", [(c, t) for c, ts in SPEC_MARKERS.items() for t in ts])
def test_spec_markers_are_single_added_tokens(inventory, assert_parity, category, token):
    by_content = {t.content: t for t in inventory.added_tokens}
    assert token in by_content, f"{token} missing from added_tokens ({category})"
    assert assert_parity(token) == [by_content[token].id]


@pytest.mark.parametrize("text", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_scenarios(assert_parity, text):
    assert_parity(text)


_REGULAR = [t for t in _inv.added_tokens if t.id not in CONTROL_IDS]


@pytest.mark.parametrize("token", _REGULAR, ids=_case_ids(_REGULAR))
def test_added_token_alone(assert_parity, token):
    assert assert_parity(token.content) == [token.id]


@pytest.mark.parametrize("token", _REGULAR, ids=_case_ids(_REGULAR))
def test_added_token_in_context(assert_parity, token):
    ids = assert_parity(f"Hello {token.content} world\n{token.content}{token.content}")
    assert ids.count(token.id) >= 2 or token.content.isspace()


@pytest.mark.parametrize("token", _inv.control_tokens, ids=_case_ids(_inv.control_tokens))
def test_control_token_parity_without_sentencepiece(assert_parity, token):
    assert assert_parity(token.content, NON_SP) == [token.id]
    assert_parity(f"x {token.content} y", NON_SP)


@pytest.mark.xfail(
    strict=True, reason="SentencePiece never matches CONTROL/UNKNOWN pieces in raw text"
)
@pytest.mark.parametrize("token", _inv.control_tokens, ids=_case_ids(_inv.control_tokens))
def test_control_token_sentencepiece_divergence(assert_parity, token):
    assert_parity(token.content)


def test_added_token_ids_decode_roundtrip(encoders, inventory):
    for t in inventory.added_tokens:
        assert encoders.decode([t.id]) == t.content
