## Describe the issue

`FusionOptimizer.apply` plans a batch of loop-fusion rewrites up front
via `find_fuseable_subgraphs` and yields them one at a time. The
generator's insertion logic
(`pytensor/tensor/rewriting/elemwise.py:836-866`) only considers
*upstream* (ancestor) dependencies when placing a new subgraph in
`sorted_subgraphs` - the `unfuseable_ancestors_bitset` check. There is
no symmetric check against the *downstream* `unfuseable_clients_bitset`
that the exploration already computes a few lines above. When a
later-discovered subgraph N happens to *produce* a variable consumed by
an earlier-listed subgraph M, the planner can place N before M.
`replace_all_validate` for N then orphans M's
`inputs`-as-blockers. When M is yielded, the
`toposort(outputs, blockers=inputs)` in
`FusionOptimizer.elemwise_to_scalar` walks straight past the dead
blockers, discovers nodes that were never part of M's planned subgraph,
and crashes with

```
File ".../pytensor/tensor/rewriting/elemwise.py", line 538, in elemwise_to_scalar
    scalar_inputs = [replacement[inp] for inp in node.inputs]
                     ~~~~~~~~~~~^^^^^
KeyError: joined_inputs
```

`SequentialGraphRewriter.apply` catches the exception and its default
failure callback logs the traceback at `ERROR` level then continues.
So the half-completed fusion is abandoned, compilation finishes, and
the produced function is numerically correct (verified to ~12 decimal
places). The visible effect is a 3-line `ERROR` log emitted from
`pytensor.graph.rewriting.basic` during `pytensor.function`, looking
like a fatal traceback even though it isn't.

This is a regression from
#1615, which shipped the
bitset-based planner in 3.0.2. The variable name `joined_inputs` in
the traceback is just the name of the reproducer's input - any
pytensor input that happens to be the first unblocked leaf reached by
the detoured `toposort` would surface there.

## Reproducible code example

Pure pytensor - no pymc / nutpie / scan / anything else needed:

```python
import logging
import pytensor
import pytensor.tensor as pt

records = []
class _Capture(logging.Handler):
    def emit(self, record):
        records.append((record.levelname, record.getMessage()))
logging.getLogger("pytensor.graph.rewriting.basic").addHandler(_Capture())

x = pt.vector("joined_inputs", shape=(2,))
a, b = x[0:1], x[1:2]
T = pt.as_tensor([0.0, 1.0])


def _stable_logsumexp_vec1(z):
    """log(sum(exp(z))) via Max-subtraction; returns shape (1,)."""
    mv = pt.expand_dims(pt.max(z), 0)
    sw = pt.switch(pt.isinf(mv), pt.exp(mv), pt.exp(z - mv))
    return mv + pt.log(pt.expand_dims(pt.sum(sw), 0))


def _stable_logsumexp_scalar(z):
    """Same numerical recipe, but returning a scalar."""
    m = pt.max(z)
    mv = pt.expand_dims(m, 0)
    sw = pt.switch(pt.isinf(mv), pt.exp(mv), pt.exp(z - mv))
    return m + pt.log(pt.sum(sw))


# Compute the same log-sum-exp twice in independent Python expressions -
# the two structurally-distinct return shapes prevent MergeOptimizer
# from folding them into a single computation.
floor_v = _stable_logsumexp_vec1(a + T)
floor_s = _stable_logsumexp_scalar(a + T)

mu = floor_v + 2 * pt.softplus((b - floor_v) / 2)
total = (-0.5 * pt.pow(T - mu, 2)).sum() - pt.softplus((floor_s - b.squeeze()) / 2)
grad = pytensor.grad(total, x)

pytensor.function([x], [total, grad])

for lvl, msg in records:
    print(f"[{lvl}] {msg.splitlines()[0]}")
```

Each of the following changes alone suppresses the bug:

- replace `x[0:1]` / `x[1:2]` with `x[0]` / `x[1]` (canonicalises to
  `Subtensor{i}` instead of `Subtensor{start:stop}`, and the planner
  takes a different path);
- drop the `pt.switch(pt.isinf(mv), pt.exp(mv), pt.exp(z - mv))`
  branch (use naive `pt.log(pt.sum(pt.exp(z)))` or the Max-subtraction
  form without the `Switch`);
- compute `floor` once and reuse - either with both calls returning
  the same shape, or with `floor_s = floor_v.squeeze()`;
- replace `T = [0.0, 1.0]` with a vector whose entries are equal -
  constant folding short-circuits the Max stable form;
- drop the `softplus(... / 2)` on either side (the `2`/`1/2` pair is
  algebraically inert when k=1, so the bug disappears at k=1);
- drop `pytensor.grad` and compile just `[total]` (the gradient is
  what makes the fused subgraphs dense enough for the planner to
  reorder them incorrectly).

## Error message

```
[ERROR] SequentialGraphRewriter apply <pytensor.tensor.rewriting.elemwise.FusionOptimizer object at 0x...>
[ERROR] Traceback:
[ERROR] Traceback (most recent call last):
  File ".../pytensor/graph/rewriting/basic.py", line 289, in apply
    sub_prof = rewriter.apply(fgraph)
               ^^^^^^^^^^^^^^^^^^^^^^
  File ".../pytensor/tensor/rewriting/elemwise.py", line 886, in apply
    scalar_inputs, scalar_outputs = self.elemwise_to_scalar(inputs, outputs)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../pytensor/tensor/rewriting/elemwise.py", line 538, in elemwise_to_scalar
    scalar_inputs = [replacement[inp] for inp in node.inputs]
                     ~~~~~~~~~~~^^^^^
KeyError: joined_inputs
```

## PyTensor version information

```
pytensor 3.0.2
python   3.12
platform Linux x86_64
```

`pytensor 3.0.2` ships the new bitset planner from #1615 (merged
2025-09-30); the bug is not present in the previous fusion
implementation.

## Context for the issue

Two viable directions for a fix:

1. **Defensive guard at the call site.** Before
   `self.elemwise_to_scalar(inputs, outputs)` in
   `FusionOptimizer.apply`, check that every `inp` in `inputs` still
   has `fgraph.clients[inp]`. If any input is dead, `continue` - its
   role as a toposort blocker has been invalidated by an earlier
   fusion in this same `apply` invocation. The next call will re-plan
   against the updated graph. One-line patch; verified on the
   reproducer that it turns 3 `ERROR` records into 0 with no change in
   the computed function:

   ```python
   _fg_clients = fgraph.clients
   if any(not _fg_clients.get(inp) for inp in inputs):
       continue
   ```

2. **Fix the ordering in `find_fuseable_subgraphs`.** When inserting a
   new subgraph N, also ensure N comes *after* any already-present
   subgraph M whose inputs include any of N's outputs. The symmetric
   counterpart of the existing `unfuseable_ancestors_bitset` check;
   `unfuseable_clients_bitset` is already computed during N's
   exploration but is currently discarded once the subgraph is closed.
   This addresses the root cause; (1) is a localised backstop that
   would still be reasonable as belt-and-braces.

Happy to send a PR for (1) or to prepare (2) with the corresponding
test from the reproducer.
