"""Fixed-vocab byte-level state serialization (prereg §5.1).

The vocabulary is FIXED and corpus-independent, deliberately -- it never
grows or shifts based on what appears in the corpus (variable names, type
names, value contents), which is what makes it leakage-proof: no token id
here can only exist because some particular corpus sample happened to
produce it. Every non-marker id is a raw byte (0..255); everything the
corpus contributes is spelled out byte-by-byte through that same fixed
256-wide alphabet, never through a corpus-derived id of its own.

Layout:
    0..255      raw byte values (any utf-8-encoded byte lands here)
    256         PAD
    257         BOS
    258         EOS
    259         KEY    (marks the start of a local's name)
    260         TYPE   (marks the start of a local's type_name)
    261         VAL    (marks the start of a local's value_repr)
    262..325    LINE_BASE + bucket, bucket = min(line, 63) -- 64 buckets,
                clamping any line number past 63 into the last bucket
                rather than growing the vocabulary or raising.

VOCAB_SIZE = 262 + 64 = 326.
"""
from __future__ import annotations

from typing import Sequence

from crucible.latent.config import INPUT_CHAR_CAP, MAX_SNAPSHOT_TOKENS, STATE_VALUE_CAP
from crucible.latent.harvest import Snapshot

PAD = 256
BOS = 257
EOS = 258
KEY = 259
TYPE = 260
VAL = 261
LINE_BASE = 262

_NUM_LINE_BUCKETS = 64
_MAX_LINE_BUCKET = _NUM_LINE_BUCKETS - 1

VOCAB_SIZE = LINE_BASE + _NUM_LINE_BUCKETS


def encode_snapshot(s: Snapshot) -> list[int]:
    """One Snapshot -> fixed-vocab token ids.

    `[LINE_BASE + min(s.line, 63)]` followed by, for each local IN THE
    ORDER GIVEN in `s.locals` (Task 1's harvest already name-sorts them;
    this function does not re-sort them and does not assert that they are
    sorted -- it trusts the order it is handed):
    `KEY + utf8(name) + TYPE + utf8(type_name) + VAL + utf8(value_repr[:24])`.

    The full sequence is built first and only THEN truncated to
    `config.MAX_SNAPSHOT_TOKENS` -- so the leading LINE token is always
    present in the output, and the cap can only ever drop whole trailing
    tokens off the very end, never reorder or half-write a local's own
    marker/byte pairing mid-construction.
    """
    line_bucket = min(s.line, _MAX_LINE_BUCKET)
    tokens: list[int] = [LINE_BASE + line_bucket]
    for name, type_name, value_repr in s.locals:
        tokens.append(KEY)
        tokens.extend(name.encode("utf-8"))
        tokens.append(TYPE)
        tokens.extend(type_name.encode("utf-8"))
        tokens.append(VAL)
        tokens.extend(value_repr[:STATE_VALUE_CAP].encode("utf-8"))
    return tokens[:MAX_SNAPSHOT_TOKENS]


def encode_input(args_literal: str) -> list[int]:
    """The call's args literal -> `[BOS] + utf8(args_literal[:96]) + [EOS]`."""
    tokens: list[int] = [BOS]
    tokens.extend(args_literal[:INPUT_CHAR_CAP].encode("utf-8"))
    tokens.append(EOS)
    return tokens


def encode_state_sequence(
    snapshots: Sequence[Snapshot], max_snapshots: int
) -> list[list[int]]:
    """The first `max_snapshots` snapshots, each run through `encode_snapshot`.

    This function only slices `snapshots` to `max_snapshots` and encodes
    what remains -- it does not count, flag, or report whether that slice
    actually dropped anything. Counting that truncation is the CALLER's
    job, same as harvest.py counts its own MAX_SNAPSHOTS truncation.
    """
    return [encode_snapshot(s) for s in snapshots[:max_snapshots]]
