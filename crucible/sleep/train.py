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

*Seeded, not merely reproducible-in-principle.* ``train`` calls
``transformers.set_seed(seed)`` before building the trainer so a repeated call with the
same ``pairs``/``seed`` is as deterministic as PyTorch's own determinism guarantees allow
-- the same discipline as ``sft_pairs``'s explicit sort, extended to the one place in
this package that touches a model's random initialisation and sampling.

*Loss scope: full-sequence, not completion-only (implementer choice, recorded per the
task brief).* TRL's completion-only masking (``DataCollatorForCompletionOnlyLM``) keys
off a literal ``response_template`` string boundary in the *rendered* prompt+completion
text, which presumes a chat template the base model has been instruction-tuned with a
stable delimiter for. ``BASE_MODEL`` does have one, but pinning the mask to it is a
second identity assumption on top of ``BASE_MODEL`` itself, entirely separate from
anything this seam's callers need to agree on -- and getting the delimiter wrong is a
silent-mismask failure mode (the loss would silently train on the wrong span), not a loud
one. Full-sequence loss (train on prompt+completion concatenated, no masking) has no such
failure mode: it is the simpler, harder-to-get-wrong choice, at the cost of also
back-propagating through the prompt tokens. Since ``LoraTrainer`` has no unit test
(GPU-only, live-checked in a later task), "harder to get wrong" was weighted over "more
principled" for a component this seam's automated tests cannot themselves catch a
mistake in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

# The A2 proposer (spec §2 amendment A2; see crucible/run/arm.py, crucible/run/serving.py)
# -- the frozen base every LoRA adapter this package trains is attached to.
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LORA_RANK = 16
LORA_ALPHA = 16
# Attention + MLP projection leaves to adapt -- the same list scripts/lora_attach_smoke.py
# targets, minus its vision-tower exclusion (BASE_MODEL is a text-only causal LM, so there
# is no "vis" leaf to filter here). Named so both trainers this package ever grows agree
# on one list rather than each re-deriving it.
LORA_TARGET_PROJ_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


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

    def __init__(self, model_name: str = BASE_MODEL, *, max_seq_length: int = 2048) -> None:
        self._model_name = model_name
        self._max_seq_length = max_seq_length

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

        # Full-sequence loss (see module docstring): each example is the prompt and its
        # landed module concatenated as one training text, no completion-only masking.
        texts = [f"{prompt}\n{module}" for prompt, module in pairs]
        dataset = Dataset.from_dict({"text": texts})

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

        config = SFTConfig(
            output_dir=str(out_dir), seed=seed, max_seq_length=self._max_seq_length,
            dataset_text_field="text", packing=False, report_to=[],
        )
        trainer = SFTTrainer(model=peft_model, args=config, train_dataset=dataset, processing_class=tokenizer)
        trainer.train()

        peft_model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        return out_dir
