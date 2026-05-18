"""Minimal pure-pytensor reproducer for the FusionOptimizer KeyError.

Run with::

    uv venv --python 3.12
    uv pip install "pytensor==3.0.2"

Expected output: 3 ``ERROR`` records from ``pytensor.graph.rewriting.basic``
whose third line's traceback ends in::

    File ".../pytensor/tensor/rewriting/elemwise.py", line 538, in elemwise_to_scalar
      scalar_inputs = [replacement[inp] for inp in node.inputs]
                       ~~~~~~~~~~~^^^^^
    KeyError: joined_inputs

``pytensor.function`` returns a perfectly usable compiled function regardless
- ``SequentialGraphRewriter`` catches the exception, logs it at ``ERROR``
level, and skips that one fusion. The error is purely a noisy log.

The variable name ``joined_inputs`` is incidental - it's just the name of
this script's input. Any pytensor input that happens to be the first
unblocked leaf reached by ``toposort``'s detour through the post-fusion
graph would produce the same crash with that variable's name.

Necessary ingredients (each removable change suppresses the bug):

    - Two scalars sliced out of a length-2 vector. The slice has to go
      through ``Reshape``/``DimShuffle``/``Sum`` (e.g.
      ``x[0:1].reshape(())`` or ``x[0:1].squeeze()``); a direct
      ``x[0]`` (which canonicalises to ``Subtensor{i}``) does not
      trigger.
    - ``floor`` computed via the *stable* form of ``log(sum(exp(...)))``
      with the ``Switch(Isinf, exp(max), exp(z - max))`` branch -
      ``log(sum(exp(...)))`` without the Max-subtraction, or with the
      subtraction but without the ``Switch``, does not trigger.
    - The ``floor`` computed *twice* in independent Python expressions,
      one returning a vector of shape ``(1,)`` and one returning a
      scalar. With both calls structurally identical pytensor's
      ``MergeOptimizer`` collapses them into a single computation, and
      the bug disappears.
    - A vector ``T`` with two distinct entries inside the ``logsumexp``
      argument. Duplicating entries lets constant folding short-circuit
      the Max stable form.
    - A vector with distinct entries inside ``pow(T - mu, 2).sum()``
      (same reason - reused as the "observed" data here).
    - Both a ``softplus((b - floor) / 2)`` term (consuming ``floor`` as
      a vector) and a ``softplus((floor - b) / 2)`` term (consuming the
      scalar floor). Either side alone is fine.
    - ``pytensor.grad`` of the combined scalar. Compiling just
      ``[total]`` (no gradient) is fine.

The bug is in pytensor's ``FusionOptimizer.apply``: its eager planner
inside ``find_fuseable_subgraphs`` orders ``sorted_subgraphs`` using
only ``unfuseable_ancestors_bitset`` and never checks
``unfuseable_clients_bitset``, so a later-discovered producer subgraph
can land before its earlier-discovered consumer. When the producer is
fused first, the consumer's pre-recorded ``inputs``-as-blockers are
orphaned and ``toposort`` walks past them into unrelated upstream
nodes. See ``investigation.md`` in this repo for a node-by-node trace.
"""

import logging

import pytensor
import pytensor.tensor as pt

records = []


class _Capture(logging.Handler):
    def emit(self, record):
        records.append((record.levelname, record.getMessage()))


logging.getLogger("pytensor.graph.rewriting.basic").addHandler(_Capture())

x = pt.vector("joined_inputs", shape=(2,))
a = x[0:1]
b = x[1:2]
T = pt.as_tensor([0.0, 1.0])


def _stable_logsumexp_vec1(z):
    """log(sum(exp(z))) via Max-subtraction; returns shape (1,)."""
    mv = pt.expand_dims(pt.max(z), 0)
    sw = pt.switch(pt.isinf(mv), pt.exp(mv), pt.exp(z - mv))
    return mv + pt.log(pt.expand_dims(pt.sum(sw), 0))


def _stable_logsumexp_scalar(z):
    """Same numerical recipe, but returning a scalar - structurally distinct
    enough from the vec-(1,) form that pytensor's MergeOptimizer doesn't
    fold the two into one."""
    m = pt.max(z)
    mv = pt.expand_dims(m, 0)
    sw = pt.switch(pt.isinf(mv), pt.exp(mv), pt.exp(z - mv))
    return m + pt.log(pt.sum(sw))


floor_v = _stable_logsumexp_vec1(a + T)
floor_s = _stable_logsumexp_scalar(a + T)

mu = floor_v + 2 * pt.softplus((b - floor_v) / 2)
obs_term = (-0.5 * pt.pow(T - mu, 2)).sum()
pot_term = -pt.softplus((floor_s - b.squeeze()) / 2)

total = obs_term + pot_term
total.name = "__logp"
grad = pytensor.grad(total, x)
grad.name = "Join.0"

records.clear()
f = pytensor.function([x], [total, grad])

print(f"\n--- captured {len(records)} pytensor.graph.rewriting.basic records ---")
for lvl, msg in records:
    print(f"[{lvl}] {msg.splitlines()[0]}")
