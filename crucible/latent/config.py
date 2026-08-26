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
