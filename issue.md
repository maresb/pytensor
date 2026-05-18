## Describe the issue

`FusionOptimizer.apply` plans a batch of loop-fusion rewrites up front
via `find_fuseable_subgraphs` and yields them one at a time. The
generator's insertion logic
(`pytensor/tensor/rewriting/elemwise.py:836-866`) only considers
*upstream* (ancestor) dependencies when placing a new subgraph in
`sorted_subgraphs` - the `unfuseable_ancestors_bitset` check. There is
no symmetric check against the *downstream* `unfuseable_clients_bitset`
that the exploration already computes a few lines above. When a
later-discovered subgraph N happens to *produce* a variable consumed
by an earlier-listed subgraph M, the planner can place N before M.
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
places on the reproducer below). The visible effect for a user is a
3-line `ERROR` log emitted from `pytensor.graph.rewriting.basic`
during `pm.sample`, looking like a fatal traceback even though it
isn't.

This is a regression from
#1615, which shipped the
bitset-based planner in 3.0.2. The variable name `joined_inputs` in
the traceback is incidental - it's the name pymc gives the joined NUTS
parameter vector. Any pytensor input that happens to be the first
unblocked leaf reached by the detoured toposort would produce the same
crash with that variable's name.

## Reproducible code example

```python
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
    pm.sample(draws=2, tune=2, chains=1, cores=1,
              progressbar=False, random_seed=0, nuts_sampler="pymc")

for lvl, msg in records:
    print(f"[{lvl}] {msg.splitlines()[0]}")
```

Each of the following changes alone suppresses the bug:

- drop one of the priors (need at least two so `joined_inputs` is size > 1);
- drop the `Potential`;
- drop the observed `Normal`;
- change the scaling factor `2` to `1` (then `k * softplus(x/k) == softplus(x)` collapses);
- replace the `[0.0, 1.0]` constant with a vector whose entries are equal (constant folding short-circuits the `Max` stable form);
- replace the observed `[0.0, 1.0]` with `[0.0, 0.0]` (same reason).

I was not able to reduce this to a pure-pytensor reproducer (no pymc
at runtime). Hand-built graphs that mimic pymc's logp structure
node-by-node and use pymc's own `compile()` / `rewrite_pregrad` /
`CheckParameterValue` /
`join_nonshared_inputs` machinery do not trigger the planner ordering
mistake, so some specific feature of the graph pymc passes to
`pytensor.function` is required and I couldn't narrow it further than
the model above. Captured pickles of pymc's pre-rewrite graph reproduce
deterministically when re-compiled with bare `pytensor.function`, so
the bug is squarely in pytensor's fusion planner - pymc just happens
to construct a graph that exercises the path.

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
pymc     6.0.0+2.g169e90128  (pymc-devs/pymc main as of bisect)
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
   against the updated graph. One-line patch; I verified on the
   reproducer that it turns 3 `ERROR` records into 0 with no change in
   posterior means or stds.

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
