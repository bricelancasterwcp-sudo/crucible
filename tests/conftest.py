import os

# The pyc rule: never let stale bytecode survive a mutation-check run.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
