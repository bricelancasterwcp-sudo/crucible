from crucible.proposer.prompt import build_prompt
from crucible.sandbox.runner import TestReport
from crucible.stream.units import Unit, sha256_text

SRC = "def add(a, b):\n    return a - b\n"  # mutated (bug)
U = Unit("X/0","unit_x","add",SRC,"from unit_x import add as candidate\ndef test_v0():\n    assert candidate(1,2)==3\n","h",sha256_text(SRC),1,0,())
SYM = TestReport((), ("test_v0",), (), (), 0.1, None)

def test_prompt_contains_source_tests_symptom_and_codec_instruction():
    p = build_prompt(U, SYM)
    assert "def add(a, b):" in p and "test_v0" in p
    assert "complete" in p.lower() and "```python" in p
    assert "hidden" not in p.lower()      # never leak the notion of hidden tests as runnable

def test_feedback_is_included_on_refinement():
    p = build_prompt(U, SYM, feedback="attempt 1 still failed test_v0")
    assert "attempt 1 still failed test_v0" in p


# --- A_noMem byte-identity + the S3 memory block -----------------------------
#
# The arms must differ only by the pre-registered columns, so A_noMem's prompt has to stay
# byte-for-byte what S2 sent. GOLDEN_NO_MEMORY is that exact S2-era text, captured before
# ``memory`` existed and written out here as a literal (not rebuilt from the template, which
# would drift along with any bug). Any template edit, or a memory block leaking in when
# ``memory`` is None, fails here loudly.

GOLDEN_NO_MEMORY = (
    "You are repairing a single Python function. Exactly one function in the module below "
    "has been altered and is now wrong. Return the COMPLETE corrected module inside one "
    "```python block, and nothing else -- no prose, no explanation, no partial diff.\n"
    "\n"
    "## Module under repair\n"
    "```python\n"
    "def add(a, b):\n"
    "    return a - b\n"
    "```\n"
    "\n"
    "## Visible tests (run against the module above)\n"
    "```python\n"
    "from unit_x import add as candidate\n"
    "def test_v0():\n"
    "    assert candidate(1,2)==3\n"
    "```\n"
    "\n"
    "## Symptom\n"
    "The visible test suite was executed once and reported:\n"
    "failed: test_v0\n"
    "\n"
    "Now output the entire fixed module as one complete ```python ... ``` block. Reproduce "
    "every line the module needs to run -- imports, every function, all of it -- not just "
    "the line you changed. Output only that block.\n"
)

MEM = ("## Prior experience with this code\n"
       "- ARITH: a prior repair changed `a - b` to `a + b` and passed.")

INSTRUCTION_MARK = "Now output the entire fixed module"


def test_no_memory_argument_is_byte_identical_to_the_s2_prompt():
    assert build_prompt(U, SYM) == GOLDEN_NO_MEMORY


def test_explicit_memory_none_is_byte_identical_to_the_s2_prompt():
    # A_noMem is threaded with memory=None; that path must not add so much as a newline.
    assert build_prompt(U, SYM, memory=None) == GOLDEN_NO_MEMORY
    assert build_prompt(U, SYM, memory=None) == build_prompt(U, SYM)


def test_memory_block_sits_between_the_symptom_and_the_instruction():
    p = build_prompt(U, SYM, memory=MEM)
    assert MEM in p
    assert p.index("failed: test_v0") < p.index(MEM) < p.index(INSTRUCTION_MARK)


def test_memory_block_is_the_only_difference_from_the_no_memory_prompt():
    # Excising the block (and the blank line that separates it from the instruction) must
    # restore the golden text exactly: the block is additive, it rewrites nothing.
    p = build_prompt(U, SYM, memory=MEM)
    assert p.replace(MEM + "\n\n", "") == GOLDEN_NO_MEMORY


def test_memory_and_feedback_coexist_on_a_refinement_prompt():
    p = build_prompt(U, SYM, feedback="attempt 1 still failed test_v0", memory=MEM)
    assert MEM in p
    assert "attempt 1 still failed test_v0" in p
    assert p.index(MEM) < p.index(INSTRUCTION_MARK) < p.index("attempt 1 still failed")


def test_empty_or_blank_memory_is_no_memory_not_an_empty_section():
    # An empty block is *nothing retrieved*, not a section with nothing in it: emitting one
    # would drop a stray blank line into a prompt that must otherwise be A_noMem's byte for
    # byte. ``retrieve()`` returns None rather than "" today, but this template must not
    # depend on a contract enforced two files away.
    for blank in ("", "\n", "\n\n", "   ", "  \n \n"):
        assert build_prompt(U, SYM, memory=blank) == GOLDEN_NO_MEMORY, repr(blank)


def test_memory_block_edges_are_normalised_so_the_section_stands_alone():
    # Leading/trailing newlines on the retrieved block must not add blank lines around the
    # section; the block's own internal formatting is left untouched.
    assert build_prompt(U, SYM, memory="\n" + MEM + "\n\n") == build_prompt(U, SYM, memory=MEM)
