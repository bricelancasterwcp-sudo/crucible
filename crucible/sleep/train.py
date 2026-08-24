"""The trainer seam: one ``Trainer`` protocol behind which a fake and the real LoRA
trainer both sit, so the sleep loop (a later task) and its tests never need a GPU to run.

*Why the seam exists at all.* Every other module under ``crucible/sleep/`` is pure
Python operating on plain data -- no CUDA, no multi-GB model weights, importable and
testable on any box. Training an adapter is fundamentally not that: it needs a GPU, the
base model's weights, and PEFT/TRL/transformers, none of which this repo's test
environment has (or should need) installed. ``Trainer`` is the ``Protocol`` both sides of
that boundary agree on -- ``FakeTrainer`` is the unit-test double every other sleep-loop
test wires in (Tasks 10-11), ``LoraTrainer`` is the real GPU-bound implementation whose
only correctness check is a live smoke run (a later ops task), never a unit test.

*Import discipline is the seam's actual contract, not a nice-to-have.* This module MUST
import successfully with no CUDA and no torch/peft/trl/transformers installed at all --
that is what lets Tasks 10-11's test suites import ``crucible.sleep.train`` (for
``Trainer``/``FakeTrainer``) on a plain CI box. Every heavy import lives INSIDE
``LoraTrainer.train`` itself, never at module level; the test suite pins this two ways --
a plain import of this module in an environment with none of those four packages
installed, and a static ``ast`` parse of this file's own source asserting none of them
appear in a module-level ``import``/``from ... import`` statement, so a future edit that
hoists one of them to the top cannot slip past a docstring promise alone.

*Cumulative retrain from BASE, every call (spec §5).* ``LoraTrainer.train`` always starts
from the frozen base checkpoint (``BASE_MODEL``) and attaches a fresh LoRA adapter -- it
never loads a previous adapter and continues training it. This matches ``select.py``'s
``sft_pairs`` being cumulative rather than incremental: the whole point of retraining
from BASE over the full verified-episode history each sleep cycle is that the regression
gate (a later task) is always evaluating "does the adapter trained on everything we
currently know regress anything," never "did the last incremental step regress
something," which would be a different and much harder question to keep honest.

*Train/serve parity: conversational data, fenced completions, completion-only loss
(fix, review finding 1).* ``BASE_MODEL`` is CHAT-served in production (``crucible.run.arm``'s
``chat=True``; ``VLLMProposer`` sends ``messages=[{"role": "user", "content": prompt}]``
and lets the server apply the model's chat template -- see ``crucible.proposer.client``),
and every completion the codec (``crucible.proposer.codec.extract_module``) accepts is the
full corrected module inside ONE fenced python block (``MODULE_FENCE``, ``crucible.proposer.prompt``'s
codec, spec S4.4). An earlier version of this trainer built a raw ``f"{prompt}\n{module}"``
text blob and trained full-sequence loss over it -- a shape the model never sees at
inference (no chat template, no fence) and that risked teaching the adapter to unlearn the
one wire-format the serving codec depends on. ``sft_records`` fixes both problems at once:
it builds TRL's conversational ``{"prompt": [...], "completion": [...]}`` shape (list-of-role
dicts) so ``SFTTrainer`` renders both sides through the tokenizer's own chat template --
matching the serve path exactly, never a hand-rolled delimiter -- and it wraps every
completion in the exact fence ``extract_module`` parses (see ``sft_records``'s own
docstring), so what the adapter is trained to emit is byte-identical in shape to a real
landing submission, not a bare module dump. ``LoraTrainer.train`` then sets
``SFTConfig(assistant_only_loss=True)`` (current TRL API, v1.0.0+; NOT the older
``DataCollatorForCompletionOnlyLM``/``response_template`` mechanism, which keys off a
literal string boundary in already-rendered text and is a second, error-prone identity
assumption on top of ``BASE_MODEL``'s chat template) -- masking the loss to the assistant
turn is a one-line config flag once the data is conversational, so there is no longer a
harder-to-get-wrong-vs-more-principled tradeoff to record: conversational input plus
``assistant_only_loss=True`` is both the correct shape AND the simpler code path.

*Seeded, not merely reproducible-in-principle.* ``train`` calls
``transformers.set_seed(seed)`` before building the trainer so a repeated call with the
same ``pairs``/``seed`` is as deterministic as PyTorch's own determinism guarantees allow
-- the same discipline as ``sft_pairs``'s explicit sort, extended to the one place in
this package that touches a model's random initialisation and sampling.

*``sft_records`` is a pure, torch-free module-level function on purpose.* It is the one
piece of ``LoraTrainer``'s logic this test suite CAN pin without a GPU: no heavy import,
so it is unit-tested directly, including a literal round-trip through
``crucible.proposer.codec.extract_module`` -- proof that the fence this module writes on
the train side is the exact fence the serve side's codec parses, not merely "looks
plausible."
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from ..proposer.prompt import MODULE_FENCE  # "```python" -- the one serve-side fence; see above

# The A2 proposer (spec §2 amendment A2; see crucible/run/arm.py, crucible/run/serving.py)
# -- the frozen base every LoRA adapter this package trains is attached to.
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LORA_RANK = 16
LORA_ALPHA = 16
# Attention + MLP projection leaves to adapt. Overlaps scripts/lora_attach_smoke.py's
# _PROJ_KEYS but is not identical to it: BASE_MODEL is a text-only causal LM, so that
# script's "vis" vision-tower exclusion has nothing to filter here; and this list also
# drops that script's "in_proj"/"out_proj" entries outright (those name a fused-QKV or
# encoder-style attention layout Qwen2.5-Coder's decoder-only architecture does not use --
# carrying them over as dead names that would never match anything would serve no purpose).
# Named so both trainers this package ever grows agree on one list rather than each
# re-deriving it.
LORA_TARGET_PROJ_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def sft_records(pairs: list[tuple[str, str]]) -> list[dict]:
    """``(root_prompt, landed_module)`` pairs -> TRL conversational SFT records.

    Each pair becomes one ``{"prompt": [...], "completion": [...]}`` record -- TRL's
    prompt-completion conversational shape, which ``SFTTrainer`` renders through the
    model's own chat template for both sides rather than a hand-built delimiter (see the
    module docstring's train/serve-parity note). The completion is wrapped in exactly the
    fence ``crucible.proposer.codec.extract_module`` looks for (``MODULE_FENCE``, a
    newline, the module, a newline, the closing fence) so the adapter is trained to emit
    the same shape a real landing submission has. ``landed_module.rstrip("\n")``
    normalises trailing newlines before re-adding exactly one, so a module stored with any
    number of trailing newlines (this codebase's convention is exactly one) still produces
    one canonical fenced block; pure stdlib string work, no heavy import, so this function
    alone is unit-tested directly (no GPU needed) -- see ``tests/sleep/test_registry.py``.
    """
    records: list[dict] = []
    for root_prompt, landed_module in pairs:
        fenced = f"{MODULE_FENCE}\n{landed_module.rstrip('\n')}\n```"
        records.append({
            "prompt": [{"role": "user", "content": root_prompt}],
            "completion": [{"role": "assistant", "content": fenced}],
        })
    return records


class Trainer(Protocol):
    """What the sleep loop needs from a trainer: pairs + seed in, an adapter dir out."""

    def train(self, pairs: list[tuple[str, str]], seed: int, out_dir: Path) -> Path:
        """Train an adapter from ``pairs`` (root_prompt, landed_module), seeded, into ``out_dir``."""
        ...


class FakeTrainer:
    """The unit-test double (Tasks 10-11 wire this in): no model, no GPU, no delay.

    Writes ``out_dir/adapter_config.json`` with ``{"pairs": len(pairs), "seed": seed}`` --
    enough for a test to assert the seam was called with the right inputs without paying
    for a real training run -- and returns ``out_dir``, matching ``Trainer.train``'s
    contract of returning the adapter directory.
    """

    def train(self, pairs: list[tuple[str, str]], seed: int, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"pairs": len(pairs), "seed": seed}
        (out_dir / "adapter_config.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return out_dir


class LoraTrainer:
    """The real trainer: PEFT LoRA rank 16 on ``BASE_MODEL``, TRL's ``SFTTrainer``.

    No unit test -- GPU-bound; its live check is a later ops task (the hot-swap smoke).
    Every heavy import (``torch``, ``datasets``, ``peft``, ``transformers``, ``trl``)
    lives inside :meth:`train`, never at module level -- see the module docstring's
    import-discipline note.
    """

    def __init__(self, model_name: str = BASE_MODEL, *, max_length: int = 2048) -> None:
        self._model_name = model_name
        self._max_length = max_length

    def train(self, pairs: list[tuple[str, str]], seed: int, out_dir: Path) -> Path:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer

        set_seed(seed)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Conversational, fenced, completion-masked (see module docstring's train/serve
        # parity note) -- sft_records is the pure helper this module's own tests pin.
        dataset = Dataset.from_list(sft_records(pairs))

        model = AutoModelForCausalLM.from_pretrained(
            self._model_name, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        targets = sorted({
            name.split(".")[-1]
            for name, mod in model.named_modules()
            if isinstance(mod, torch.nn.Linear) and any(key in name for key in LORA_TARGET_PROJ_KEYS)
        })
        peft_model = get_peft_model(
            model, LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=targets, task_type="CAUSAL_LM")
        )

        # assistant_only_loss=True (current TRL API, v1.0.0+) masks the loss to the
        # completion turn now that the data is conversational -- see module docstring.
        # NOTE: the SFTConfig kwarg is max_length, not the older max_seq_length.
        config = SFTConfig(
            output_dir=str(out_dir), seed=seed, max_length=self._max_length,
            assistant_only_loss=True, packing=False, report_to=[],
        )
        trainer = SFTTrainer(model=peft_model, args=config, train_dataset=dataset, processing_class=tokenizer)
        trainer.train()

        peft_model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        return out_dir
