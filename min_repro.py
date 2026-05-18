"""Minimal reproducer for pytensor FusionOptimizer KeyError.

Run in a Python 3.12 venv with:
    uv pip install "pymc @ git+https://github.com/pymc-devs/pymc.git@main"

Resolves to pymc 6.0.0+2.g169e90128 / pytensor 3.0.2.

Expected behaviour:
- 3 ``ERROR`` records are captured from ``pytensor.graph.rewriting.basic``.
- The third record's traceback ends with::

    File ".../pytensor/tensor/rewriting/elemwise.py", line 538, in elemwise_to_scalar
      scalar_inputs = [replacement[inp] for inp in node.inputs]
                       ~~~~~~~~~~~^^^^^
    KeyError: joined_inputs

- ``pm.sample`` still completes; the failure is logged but swallowed by
  ``SequentialGraphRewriter`` and the rewrite is simply skipped, so
  posteriors are correct. The log is misleading, not fatal.

Variations that DO NOT reproduce (each suppresses the bug):
- Drop the ``pm.Potential`` -> no error.
- Drop the observed ``pm.Normal`` -> no error.
- Drop the second prior (``xi_u``) -> no error.
- Replace ``0.005`` with ``1.0`` (so ``k * softplus(x/k) == softplus(x)``
  trivially canonicalises away) -> no error.
- Make ``T`` length 1 (or replace it with a scalar) -> no error.

All three structural elements (two scalar priors, the ``Potential``, the
observed ``Normal``) plus a non-trivial ``k != 1`` scaling are required to
construct a graph in which ``FusionOptimizer.find_fuseable_subgraphs``
pre-plans two subgraphs in the wrong dependency order.
"""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

records = []


class _Capture(logging.Handler):
    def emit(self, record):
        records.append((record.levelname, record.getMessage()))


logging.getLogger("pytensor.graph.rewriting.basic").addHandler(_Capture())

T = pt.as_tensor([-1.0, 1.0])

with pm.Model() as model:
    b0 = pm.Normal("b0")
    xi_u = pm.Normal("xi_u")
    floor = pt.log(pt.sum(pt.exp(b0 + T)))
    xi = floor + 0.005 * pt.softplus((xi_u - floor) / 0.005)
    pm.Potential("p", -pt.softplus((floor - xi_u) / 0.005))
    pm.Normal("obs", mu=xi, observed=[1.0, 2.0])

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
