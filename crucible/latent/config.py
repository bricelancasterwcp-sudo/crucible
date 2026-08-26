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

# Target number of ACCEPTED FUNCTIONS the corpus harvest (crucible.latent.gen
# .generate_corpus) drives toward. chosen, prereg §4.
TARGET_FUNCTIONS = 5000

# Floor gate (prereg §4): crucible.latent.corpus.build_manifest stamps
# "floor_functions": "PASS" when the corpus's accepted-function count is at
# least this many, else "FAIL" -- Task 4 (ops) reads this verdict off the
# manifest, it does not recompute it. chosen, prereg §4.
FLOOR_FUNCTIONS = 3000

# Floor gate (prereg §4) on the harvest's nondeterminism-rejection rate:
# build_manifest stamps "nondet_kill": "FAIL" once the fraction of
# determinism-screened (function, input) pairs that came back nondeterministic
# exceeds this -- a harvest process too noisy to trust past this point.
# chosen, prereg §4.
NONDET_REJECT_KILL = 0.40

# Seed mixed into assign_split's hash (crucible.latent.corpus) -- fixed so a
# corpus's train/val/test partition is reproducible run over run. chosen,
# prereg §4.
SPLIT_SEED = 0

# Train/val/test fractions for assign_split's function-level hash partition
# (prereg §4). Must sum to 1.0. chosen, prereg §4.
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
