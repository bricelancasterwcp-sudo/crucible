"""RED/GREEN tests for the fixed-vocab state serialization (prereg §5.1).

Each test here is a mutation pin, per the Task 4 brief: determinism, the
literal byte layout of KEY/TYPE/VAL markers around each local (order
preserved, no re-sort), the 24-char value truncation, the 128-token cap,
the line-bucket clamp, vocab-bounds under multi-byte utf-8, and the
encode_input BOS/EOS framing + 96-char cap.
"""
from crucible.latent.harvest import Snapshot
from crucible.latent.state import (
    BOS,
    EOS,
    KEY,
    LINE_BASE,
    TYPE,
    VAL,
    VOCAB_SIZE,
    encode_input,
    encode_snapshot,
    encode_state_sequence,
)


def test_vocab_size_is_262_plus_64():
    assert VOCAB_SIZE == 326


def test_encode_snapshot_is_deterministic():
    s = Snapshot(line=3, locals=(("a", "int", "1"), ("b", "str", "x")))
    assert encode_snapshot(s) == encode_snapshot(s)


def test_encode_snapshot_preserves_given_order_and_exact_byte_layout():
    """Locals passed OUT of alphabetical order (z before a) -- this pins
    that encode_snapshot does not re-sort, and that KEY/TYPE/VAL wrap each
    local's fields in the given order (a KEY<->VAL marker swap would flip
    the second and sixth tokens of each group and fail this)."""
    s = Snapshot(line=5, locals=(("z", "str", "hello"), ("a", "int", "1")))
    expected = [
        LINE_BASE + 5,
        KEY, *b"z", TYPE, *b"str", VAL, *b"hello",
        KEY, *b"a", TYPE, *b"int", VAL, *b"1",
    ]
    assert encode_snapshot(s) == expected


def test_encode_snapshot_truncates_value_repr_to_24_bytes():
    long_value = "x" * 40
    s = Snapshot(line=0, locals=(("v", "str", long_value),))
    tokens = encode_snapshot(s)
    # Layout: [LINE, KEY, 'v', TYPE, 's','t','r', VAL, ...24 x's]
    val_index = tokens.index(VAL)
    value_bytes = tokens[val_index + 1:]
    assert len(value_bytes) == 24
    assert value_bytes == [ord("x")] * 24


def test_encode_snapshot_truncates_value_repr_by_codepoint_not_by_byte():
    """All-ASCII fixtures can't distinguish the correct `s[:24].encode()`
    (truncate the STRING to 24 codepoints, then utf-8 encode) from a
    `s.encode()[:24]` mutant (encode first, then take the first 24 BYTES) --
    on ASCII, one char is one byte, so both forms produce exactly 24 bytes.
    Multi-byte characters break that tie: a 24-CODEPOINT slice of 2/3-byte
    characters encodes to well over 24 bytes, where a 24-BYTE slice of the
    full encoding is always exactly 24 bytes by construction. The two
    outputs differ in length here, not just content, so this fixture kills
    the mutant regardless of where any byte boundary happens to fall.
    """
    multibyte_value = "θ≈π" * 20  # 60 codepoints; each char is 2-3 utf-8 bytes
    s = Snapshot(line=0, locals=(("v", "str", multibyte_value),))
    tokens = encode_snapshot(s)
    val_index = tokens.index(VAL)
    value_bytes = tokens[val_index + 1:]
    expected = list(multibyte_value[:24].encode("utf-8"))
    assert value_bytes == expected
    assert len(value_bytes) > 24  # codepoint-slice of multi-byte chars exceeds 24 bytes


def test_encode_snapshot_caps_at_128_tokens():
    locals_ = tuple((f"n{i}", "int", "0") for i in range(50))
    s = Snapshot(line=1, locals=locals_)
    tokens = encode_snapshot(s)
    assert len(tokens) == 128


def test_encode_snapshot_clamps_line_bucket_to_63():
    s = Snapshot(line=999, locals=())
    tokens = encode_snapshot(s)
    assert tokens == [LINE_BASE + 63]


def test_encode_snapshot_stays_within_vocab_on_unicode_heavy_input():
    s = Snapshot(
        line=2,
        locals=(("日本語", "ключ", "emoji😀 string with unicode π≈3.14159…"),),
    )
    tokens = encode_snapshot(s)
    assert all(0 <= t < VOCAB_SIZE for t in tokens)


def test_encode_input_frames_with_bos_and_eos():
    tokens = encode_input("(1, 2)")
    assert tokens[0] == BOS
    assert tokens[-1] == EOS
    assert tokens[1:-1] == list(b"(1, 2)")


def test_encode_input_caps_literal_at_96_chars():
    literal = "x" * 200
    tokens = encode_input(literal)
    assert tokens[0] == BOS and tokens[-1] == EOS
    assert len(tokens) - 2 == 96


def test_encode_input_caps_by_codepoint_not_by_byte():
    """Same rationale as the value_repr codepoint-vs-byte test above, for
    `encode_input`'s 96-char cap: an all-ASCII fixture can't distinguish
    `literal[:96].encode()` (correct) from `literal.encode()[:96]` (mutant),
    since both give 96 bytes for 96 ASCII chars. A multi-byte fixture
    breaks that tie by length alone.
    """
    multibyte_literal = "λμν" * 40  # 120 codepoints; each char is 2 utf-8 bytes
    tokens = encode_input(multibyte_literal)
    assert tokens[0] == BOS and tokens[-1] == EOS
    expected = list(multibyte_literal[:96].encode("utf-8"))
    assert tokens[1:-1] == expected
    assert len(expected) > 96  # codepoint-slice of multi-byte chars exceeds 96 bytes


def test_encode_state_sequence_slices_to_max_snapshots():
    snaps = [Snapshot(line=i, locals=()) for i in range(5)]
    out = encode_state_sequence(snaps, 3)
    assert len(out) == 3
    assert out == [encode_snapshot(s) for s in snaps[:3]]
