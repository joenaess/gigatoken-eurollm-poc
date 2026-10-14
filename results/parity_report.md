# Parity report

**Status: PASS** — every case asserts identical token IDs across `hf_auto`, `hf_raw`, `sp`, `gt_compat`, `gt_native`.

Generated 2026-10-14 09:04 UTC by `pytest -q`.

| Category   | Cases   | passed   | failed   | xfailed   | xpassed   | skipped   |
|------------|---------|----------|----------|-----------|-----------|-----------|
| scripts    | 52      | 52       | 0        | 0         | 0         | 0         |
| specials   | 376     | 372      | 0        | 4         | 0         | 0         |
| whitespace | 181     | 181      | 0        | 0         | 0         | 0         |
| **total**  | 609     | 605      | 0        | 4         | 0         | 0         |

`xfailed` cases are strict expected failures: raw SentencePiece does not match CONTROL/UNKNOWN pieces (`<unk>`, `<bos>`, `<eos>`, `<pad>`) written as plain text, while HF and both gigatoken paths map them to their IDs. The remaining paths are asserted equal for those tokens in separate passing tests.

## Token inventory (derived from the tokenizer)

| Quantity                          | Derived   | Release notes   |
|-----------------------------------|-----------|-----------------|
| Vocabulary size                   | 262144    |                 |
| Added tokens (`added_tokens`)     | 176       | ~204            |
| of which `special: true`          | 163       |                 |
| SentencePiece USER_DEFINED pieces | 172       |                 |
| SentencePiece CONTROL pieces      | 3         |                 |
| SentencePiece BYTE pieces         | 256       |                 |
| Whitespace-only added tokens      | 13        |                 |
| Whitespace-only BPE pieces        | 51        |                 |
| Whitespace tokens total           | 64        |                 |
| added + literal-space pieces      | 45        | ~45             |
| reachable when encoding text      | 32        |                 |

The 32 literal-space pieces (runs of U+0020, e.g. 2–32 spaces) are never emitted: spaces are escaped to `▁` before BPE, so indentation is encoded with `▁`-run pieces instead. Every encoding path agrees on this (see `test_literal_space_pieces_are_never_emitted`).
