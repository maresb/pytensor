## Describe the issue

`FusionOptimizer.apply` plans a batch of loop-fusion rewrites up front
via `find_fuseable_subgraphs` and yields them one at a time so each can
be applied to the function graph. When two of the planned subgraphs are
related as producer→consumer (an output of subgraph A is one of the
inputs of subgraph B), the planner can place the producer A *before*
the consumer B in the yield order. `replace_all_validate` then rewires
A's outputs into a new `Composite`, leaving B's pre-recorded
`inputs`-as-blockers orphaned. When B is yielded, the
`toposort(outputs, blockers=inputs)` in
`FusionOptimizer.elemwise_to_scalar` walks straight past the dead
blockers, discovers nodes that were never part of B's planned
subgraph, and crashes with

```
File ".../pytensor/tensor/rewriting/elemwise.py", line 538, in elemwise_to_scalar
    scalar_inputs = [replacement[inp] for inp in node.inputs]
                     ~~~~~~~~~~~^^^^^
KeyError: joined_inputs
```

`SequentialGraphRewriter.apply` catches the exception and the default
`SequentialGraphRewriter.warn` callback logs it at `ERROR` level then
moves on. So the partially-completed fusion is abandoned, compilation
continues, and the produced function gives correct results. The
visible effect for a user is a 3-line `ERROR` log emitted by
`pytensor.graph.rewriting.basic` during `pm.sample`, looking like a
fatal traceback even though it isn't.

I bisected to confirm this is a regression from
#1615 (the new bitset-based fusion planner shipped in 3.0.2). The
specific gap in the planner is in
`find_fuseable_subgraphs` at the insertion logic for `sorted_subgraphs`
(`pytensor/tensor/rewriting/elemwise.py:836-866`): when deciding where
to insert a newly-discovered subgraph N, it uses
`unfuseable_ancestors_bitset` to ensure N comes before its upstream
dependencies, but there is no symmetric pass to ensure N comes after
any previously-discovered subgraph that consumes one of N's outputs.
`unfuseable_clients_bitset` is computed during the exploration and
could be reused, but is discarded once the subgraph is closed.

The variable name `joined_inputs` in the traceback is just the name
pymc gives the joined NUTS parameter vector
(`pymc/pytensorf.py::join_nonshared_inputs`). It is not specific to
that variable — any pytensor input that happens to be the first
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

T = pt.as_tensor([-1.0, 1.0])

with pm.Model() as model:
    b0 = pm.Normal("b0")
    xi_u = pm.Normal("xi_u")
    floor = pt.log(pt.sum(pt.exp(b0 + T)))
    xi = floor + 0.005 * pt.softplus((xi_u - floor) / 0.005)
    pm.Potential("p", -pt.softplus((floor - xi_u) / 0.005))
    pm.Normal("obs", mu=xi, observed=[1.0, 2.0])

with model:
    pm.sample(draws=2, tune=2, chains=1, cores=1,
              progressbar=False, random_seed=0, nuts_sampler="pymc")

for lvl, msg in records:
    print(f"[{lvl}] {msg.splitlines()[0]}")
```

Removing any one of (the second prior, the `Potential`, the observed
`Normal`) suppresses the bug — all three plus a non-trivial scaling
constant (`0.005`) are needed to produce a plan where producer and
consumer subgraphs land in the wrong order.

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

`pm.sample` completes normally; the rewrite is skipped by
`SequentialGraphRewriter` and the resulting fgraph still produces
correct posteriors (verified to ~12 decimal places against a patched
build that suppresses the crash).

## PyTensor version information

```
pytensor 3.0.2
pymc     6.0.0+2.g169e90128  (pymc-devs/pymc main as of bisect)
python   3.12
platform Linux x86_64
```

`pytensor 3.0.2` includes the new bitset planner from #1615 (merged
2025-09-30); the bug is not present in the previous fusion
implementation.

## Context for the issue

Two viable directions for a fix:

1. **Defensive guard at the call site.** Before
   `self.elemwise_to_scalar(inputs, outputs)` in
   `FusionOptimizer.apply`, check that every `inp` in `inputs` still
   has `fgraph.clients[inp]`; if any input is dead, `continue`. This
   matches what `replace_all_validate` would do later anyway, and lets
   the next `FusionOptimizer.apply` invocation re-plan against the
   updated graph. One-line patch, verified to suppress the
   `ERROR` log on the reproducer without changing posterior means or
   stds.

   ```python
   _fg_clients = fgraph.clients
   if any(not _fg_clients.get(inp) for inp in inputs):
       continue
   ```

2. **Fix the ordering in `find_fuseable_subgraphs`.** When inserting a
   new subgraph N into `sorted_subgraphs`, also ensure N is placed
   *after* any already-present subgraph M such that one of N's outputs
   is one of M's inputs. This is the symmetric counterpart of the
   existing `unfuseable_ancestors_bitset` check; the
   `unfuseable_clients_bitset` already computed during N's exploration
   is exactly the right input. This addresses the root cause; (1) is a
   localised backstop that would still be reasonable as belt-and-braces.

Happy to send a PR for (1) if that's a preferred starting point, or to
prepare (2) with the corresponding test from `min_repro.py` in this
issue. The minimal reproducer and an instrumented trace showing the
producer-before-consumer plan are reproducible with the snippet above.
