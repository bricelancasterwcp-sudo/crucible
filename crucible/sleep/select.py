"""Sleep-cycle SFT selection: which verified episodes train the next adapter.

*Cumulative, not incremental (spec §5, ruling R-S3-3).* ``sft_pairs`` returns EVERY
verified episode the store has ever accumulated, not just the ones landed since the last
sleep -- each sleep cycle trains a fresh adapter from the base model
(``train.py``'s ``LoraTrainer`` docstring) over the whole verified history to date, so
there is no incremental fine-tune chain to drift or diverge. The trade a caller is making
by asking for "all of it, every time" is deliberate: retraining from BASE on the
cumulative set is the thing that keeps the regression gate (a later task) meaningful -- a
chain of incremental adapters would make "did the LAST training step regress something" a
different, harder question than "does the adapter trained on everything we know regress
anything," and the spec picks the latter.

*Deterministic order, not insertion order.* ``store.episodes(verified_only=True)``
already returns rows in insertion (rowid) order, but this module re-sorts by
``(created_at, item_id)`` explicitly rather than trusting that order to stay correct --
insertion order and creation-time order can diverge (a store rebuilt from a JSONL export,
for instance, or two episodes written in the same batch in a different sequence than
their ``created_at``), and a caller hashing the pair list (``episode_set_hash``) needs
the hash to depend only on the store's *content*, never on some other module's write
order. ``item_id`` is the tie-breaker for two episodes that share a ``created_at``
timestamp -- content-addressed and unique, so the sort is total.

*The defensive raise can't currently fire, and that's the point.* A verified episode is,
by ``crucible.memory.schema.episode_verified``'s own definition, one whose hidden suite
passed -- which only happens after a patch actually landed, so ``landed_module`` is never
``None`` on a verified row in this codebase today (``crucible.memory.distill.distill``
pins the same invariant on its own input for the same reason). ``sft_pairs`` checks
anyway and raises loudly rather than silently building a training pair out of ``None``:
if some future change to the store, the verification path, or a hand-edited JSONL import
ever produces a verified-but-unlanded row, that is a data-integrity bug this function
refuses to paper over by skipping the row or coercing it to an empty string.

``episode_set_hash`` is the identity ``registry.py``'s ``AdapterRegistry`` mints adapter
ids from: sha256 of the canonical JSON of the pair list. ``pairs`` is a plain
``list[tuple[str, str]]`` -- no dict, no keys to reorder -- so ``sort_keys=True`` has no
effect on the pair ordering itself; it is set anyway for consistency with every other
canonical-JSON hash in this codebase (``crucible.memory.schema.content_id``). Order DOES
matter to the hash: two pair lists holding the same tuples in a different order hash
differently, on purpose (see the function's own docstring).

No clock is read anywhere in this module: everything ``sft_pairs`` reads
(``created_at`` included) is already stored on the episode by an earlier caller.
"""
from __future__ import annotations

import json

from ..memory.store import MemoryStore
from ..stream.units import sha256_text


def sft_pairs(store: MemoryStore) -> list[tuple[str, str]]:
    """``(root_prompt, landed_module)`` for every verified episode, ever -- cumulative, sorted.

    Order is ``(created_at, item_id)`` ascending, independent of the store's own row
    order. Raises ``ValueError`` if a verified episode reaches here with
    ``landed_module is None`` -- see the module docstring's "defensive raise" note; this
    cannot happen for a real ``episode_verified`` row today, so tripping it means some
    upstream invariant broke.
    """
    episodes = sorted(store.episodes(verified_only=True), key=lambda ep: (ep.created_at, ep.item_id))
    pairs: list[tuple[str, str]] = []
    for ep in episodes:
        if ep.landed_module is None:
            raise ValueError(
                f"sft_pairs refuses a verified episode with no landed_module: {ep.item_id} "
                f"(task_key={ep.task_key!r})"
            )
        pairs.append((ep.root_prompt, ep.landed_module))
    return pairs


def episode_set_hash(pairs: list[tuple[str, str]]) -> str:
    """sha256 of the canonical JSON of ``pairs`` -- the adapter identity ``registry.py`` mints from.

    The same pair list (same content, same order) always hashes to the same value; a
    different verified-episode set -- one more episode, one different landed module, or
    the same episodes in a different order -- always hashes to a different one.
    """
    canonical = json.dumps(pairs, sort_keys=True)
    return sha256_text(canonical)
