"""Minimal reproducer for pytensor FusionOptimizer KeyError.

Run with::

    uv venv --python 3.12
    uv pip install "pymc @ git+https://github.com/pymc-devs/pymc.git@main"

which resolves to pymc 6.0.0+2.g169e90128 / pytensor 3.0.2.

Expected output: 3 ``ERROR`` records from ``pytensor.graph.rewriting.basic``
whose traceback ends in::

    File ".../pytensor/tensor/rewriting/elemwise.py", line 538, in elemwise_to_scalar
      scalar_inputs = [replacement[inp] for inp in node.inputs]
                       ~~~~~~~~~~~^^^^^
    KeyError: joined_inputs

``pm.sample`` completes normally; the rewrite is caught by
``SequentialGraphRewriter`` and skipped, so posteriors are still correct.
The error is purely a noisy log.

Necessary ingredients (each removable change suppresses the bug):
    - Two parameters (``pm.Flat`` works; no log-prob needed)
    - A shared ``floor`` expression formed via ``logsumexp`` over a vector
    - Both a ``Potential`` and an observed ``Normal`` that consume ``floor``
    - Scaling factor != 1 in ``floor + k * softplus((b - floor) / k)``
      (k=1 collapses algebraically and the bug doesn't appear)
    - Distinct values in the vector constant added inside ``logsumexp``
      (duplicates let constant folding short-circuit the Max stable form)
    - Distinct values in the observed array (same reason)

I could not reproduce this with pure pytensor (no pymc at runtime) despite
matching pymc's graph structurally - some specific feature of the graph
pymc passes to ``pytensor.function`` is required and I was unable to
narrow it further than the model below. The graph differences I found are
captured in ``investigation.md``.
"""

import logging

import pymc as pm
import pytensor.tensor as pt

records = []


class _Capture(logging.Handler):
    def emit(self, record):
        records.append((record.levelname, record.getMessage()))


logging.getLogger("pytensor.graph.rewriting.basic").addHandler(_Capture())

with pm.Model() as model:
    a = pm.Flat("a")
    b = pm.Flat("b")
    floor = pt.logsumexp(a + pt.as_tensor([0.0, 1.0]))
    pm.Potential("p", -pt.softplus((floor - b) / 2))
    pm.Normal("obs", mu=floor + 2 * pt.softplus((b - floor) / 2), observed=[0.0, 1.0])

with model:
    pm.sample(
        draws=2,
        tune=2,
        chains=1,
        cores=1,
        progressbar=False,
        random_seed=0,
        nuts_sampler="pymc",
    )

print(f"\n--- captured {len(records)} pytensor.graph.rewriting.basic records ---")
for lvl, msg in records:
    print(f"[{lvl}] {msg.splitlines()[0]}")
