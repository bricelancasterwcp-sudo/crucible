"""Harvest execution knobs (prereg §4).

These are the STARTING values for the B-lite corpus harvest -- small
enough that a tiny pure-Python sample function finishes in well under a
second, generous enough that legitimate work (list/dict comprehensions,
short loops) is not mistaken for a hang.
"""

# Wall-clock budget for one `sensorium run` subprocess. chosen, prereg §4.
EXEC_TIMEOUT_S = 3.0

# RLIMIT_AS (virtual address space) cap for the recorded subprocess, in
# MiB. chosen, prereg §4.
EXEC_RLIMIT_AS_MB = 512

# Hard cap on the number of per-line locals snapshots kept per harvest.
# Exceeding it does not fail the harvest -- it sets `truncated=True` and
# keeps the first MAX_SNAPSHOTS in execution order. chosen, prereg §4.
MAX_SNAPSHOTS = 32

# Hard cap on a candidate function's total AST node count (ast.walk over the
# whole module). A generated function above this is rejected by the corpus
# validator with reason "node-count-exceeded" -- a crude but cheap guard
# against a candidate that technically parses but is far outside the
# "short, self-contained function" the corpus is built from. chosen,
# prereg §4.
MAX_AST_NODES = 400

# Binary-outcome balance guard (spec §4: "more skewed than 80/20"). Once the
# corpus has accepted enough samples, further samples of the majority binary
# class are rejected by rejection sampling while the running majority-class
# fraction exceeds this limit -- see crucible.latent.gen.generate_corpus.
# chosen, prereg §4.
SKEW_LIMIT = 0.80
