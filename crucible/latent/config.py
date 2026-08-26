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

# Hard cap on the number of tokens crucible.latent.state.encode_snapshot
# emits for one Snapshot. The full [LINE token, then per-local
# KEY/name/TYPE/type_name/VAL/value_repr groups] sequence is built first and
# THEN sliced to this length -- so truncation only ever drops whole trailing
# tokens off the end, never reorders or half-writes a local's marker/byte
# pairing. chosen, prereg §5.1.
MAX_SNAPSHOT_TOKENS = 128

# ---- B-lite model dims (crucible.latent.model), chosen at lock, prereg §5.2 ----

# Shared hidden width the frozen code encoder, the trained StateEncoder's
# projection, and the LatentPredictor all operate in. chosen, prereg §5.2.
D_MODEL = 768

# StateEncoder's internal transformer width, BEFORE the Linear(STATE_ENC_D ->
# D_MODEL) projection -- deliberately narrower than D_MODEL since a snapshot's
# fixed 326-token vocabulary (state.py) carries far less information per
# token than the frozen code encoder's subword vocabulary. chosen, prereg §5.2.
STATE_ENC_D = 512

# StateEncoder's TransformerEncoder depth. chosen, prereg §5.2.
STATE_ENC_LAYERS = 4

# LatentPredictor's TransformerEncoder depth -- the ~100M-param,
# EB-JEPA-shaped causal predictor over [z_code, z_input, z_s1..z_sT]
# (prereg §3's arm description). chosen, prereg §5.2.
PRED_LAYERS = 12

# LatentPredictor's attention head count. chosen, prereg §5.2.
PRED_HEADS = 12

# Weight on the LeWM-style isotropy regularizer term in blite_loss (prereg
# §3: "prediction + isotropic-Gaussian regularizer, no EMA, no stop-grad").
# chosen, prereg §5.2.
LAMBDA_ISO = 0.1

# GroundedHead's multiclass output width: {pass-return, exception, timeout}
# (prereg §4). The binary head (clean-return vs not) is the gating target;
# this multiclass head is the descriptive/auxiliary one. chosen, prereg §5.2.
N_OUTCOME_CLASSES = 3

# ---- B-lite training harness (crucible.latent.train), chosen at lock, prereg §5.2 ----

# AdamW learning rate. chosen, prereg §5.2.
LR = 3e-4

# Training batch size (samples per optimizer step). chosen, prereg §5.2.
BATCH = 64

# Hard cap on total optimizer steps for one train_blite() run -- the loop
# stops here even if early stopping (PATIENCE) never triggers. chosen,
# prereg §5.2.
MAX_STEPS = 20000

# Run one val-set evaluation (grounded AUROC + collapse probes, prereg §5.5)
# every this many optimizer steps. chosen, prereg §5.2.
EVAL_EVERY = 500

# Early-stop patience, in EVAL_EVERY-spaced evaluations: training stops once
# this many CONSECUTIVE evals fail to strictly improve the best val AUROC
# seen so far (a flat AUROC counts as non-improving -- see train.py). chosen,
# prereg §5.2.
PATIENCE = 5

# Seed mixed into torch/numpy/random's global RNG state at the start of
# train_blite() -- also what the per-epoch train-batch shuffle order derives
# from, so the same seed reproduces the exact same sequence of batches.
# chosen, prereg §5.2.
TRAIN_SEED = 0
