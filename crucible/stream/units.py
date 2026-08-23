"""A ``Unit``: one EvalPlus problem prepared for mutation, plus the naming it needs.

Two things here are load-bearing beyond their size.

*Docstring stripping.* The canonical solution's docstring **is** the problem
specification. A proposer that can read it is not searching, it is transcribing, so
``strip_docstrings`` removes module, class and function docstrings before the source
ever reaches a proposer. It rewrites through ``ast.unparse``, which also normalises
formatting -- so ``src_hash`` keys on semantics-as-unparsed, not on whitespace.

*Module naming.* ``module_name_for`` always prefixes ``unit_``. That is not cosmetic:
a bare EvalPlus name could land on a stdlib module (``socket``, ``json``), and a unit
file shadowing a stdlib module inside the sandbox workdir would silently disable the
network shim that imports it. The prefix also keeps the name clear of the runner's
reserved set (``crucible/sandbox/runner.py``), which raises rather than guessing.

Hashes here are content hashes -- ``sha256`` of the text -- never descriptions of it.
"""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Unit:
    """One prepared problem: stripped source, its tests, and the hashes that key them.

    Field order is frozen: downstream tasks construct ``Unit`` positionally.
    ``visible_test_src`` / ``hidden_test_src`` are filled by the test builders;
    ``dropped_inputs`` records ``(input index as "v3"/"h17", reason)`` for every
    EvalPlus input a builder refused to emit, so a thinned test set is visible in the
    record rather than inferred from a count.
    """

    unit_id: str
    module_name: str
    entry_point: str
    module_src: str
    visible_test_src: str
    hidden_test_src: str
    src_hash: str
    n_visible: int
    n_hidden: int
    dropped_inputs: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict:
        """JSON-ready form: tuples become lists so a round-trip through a file is exact."""
        d = asdict(self)
        d["dropped_inputs"] = [list(x) for x in self.dropped_inputs]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Unit":
        """Inverse of :meth:`to_dict`; restores the tuple shape so equality holds."""
        d = dict(d)
        d["dropped_inputs"] = tuple(tuple(x) for x in d["dropped_inputs"])
        return cls(**d)


def sha256_text(s: str) -> str:
    """Content hash of ``s`` -- the only kind of key this codebase mints."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def module_name_for(task_id: str) -> str:
    """``"HumanEval/0"`` -> ``"unit_humaneval_0"``. Keep the prefix; see module docstring."""
    return "unit_" + re.sub(r"[^a-z0-9]+", "_", task_id.lower()).strip("_")


class _DocstringStripper(ast.NodeTransformer):
    """Drops the leading string expression from every scope that can hold one."""

    def _strip(self, node):
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            # A body that was *only* a docstring still needs a statement to stay parseable.
            body = body[1:] or [ast.Pass()]
            node.body = body
        self.generic_visit(node)
        return node

    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _strip


def strip_docstrings(src: str) -> str:
    """Return ``src`` with every module/class/function docstring removed.

    Pure function of its input: parse, transform, ``ast.unparse``. No randomness, no
    dependence on the environment, so the same source always yields the same text and
    therefore the same ``src_hash``.
    """
    tree = _DocstringStripper().visit(ast.parse(src))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"
