# Investigation: pytensor `FusionOptimizer` `KeyError: joined_inputs`

## Summary

`FusionOptimizer.apply` in pytensor 3.0.2 plans a batch of loop-fusion
rewrites up front and then yields them one at a time so each can be
applied to the function graph. The planner's insertion logic for
`sorted_subgraphs` only considers *upstream* (ancestor) dependencies; it
never checks whether a newly-inserted subgraph would be placed before
one of its *downstream* consumers. When that happens, the producer
gets fused first, its outputs get rewired into a new Composite, and
the still-pending consumer subgraph's pre-recorded `inputs`-as-blockers
are now orphaned. `toposort(outputs, blockers=inputs)` walks straight
past those dead blockers, discovers nodes that were never part of the
consumer's planned subgraph, and crashes when those nodes' inputs are
missing from the `replacement` dict.
`SequentialGraphRewriter` catches the `KeyError`, logs it at `ERROR`
level, and skips that one rewrite, so the symptom is purely a noisy
log, not a wrong answer.

I could not find a prior issue for this. Closest:
- [#741](https://github.com/pymc-devs/pytensor/issues/741) (closed) -
  a different exception type in the *old* FusionOptimizer
  implementation.
- [#1615](https://github.com/pymc-devs/pytensor/pull/1615) (merged
  2025-09-30, ships in 3.0.2) - the PR that introduced the current
  planner. The regression lives here.
- [#249](https://github.com/pymc-devs/pytensor/issues/249) (open) -
  unrelated fusion limitation about subgraphs that share inputs.

## Failing site

```python
# pytensor/tensor/rewriting/elemwise.py
@staticmethod
def elemwise_to_scalar(inputs, outputs):
    replacement = {
        inp: get_scalar_type(inp.type.dtype).make_variable() for inp in inputs
    }
    for node in toposort(outputs, blockers=inputs):
        scalar_inputs = [replacement[inp] for inp in node.inputs]   # <-- line 538
        ...
```

`replacement` is seeded only from the subgraph's `inputs`. The
`KeyError` says `toposort` yielded a node whose input is *neither* an
`inputs` blocker *nor* an output of an earlier yielded node - i.e.
`(inputs, outputs)` is not a valid frontier for the graph that
`toposort` is actually walking. The repr "joined_inputs" is just the
name pymc gives the joined NUTS parameter vector
([`pymc/pytensorf.py:579`](https://github.com/pymc-devs/pymc/blob/main/pymc/pytensorf.py)) -
nothing about the variable itself is special; any pytensor input that
becomes the unintended toposort sink would produce the same crash.

## Mechanism

`FusionOptimizer.apply` is structured as:

```python
for inputs, outputs in find_fuseable_subgraphs(fgraph):  # generator
    scalar_inputs, scalar_outputs = self.elemwise_to_scalar(inputs, outputs)
    composite_outputs = Elemwise(Composite(...))(*inputs, return_list=True)
    fgraph.replace_all_validate(zip(outputs, composite_outputs), reason=...)
```

`find_fuseable_subgraphs` is **eager**: its entire body runs before the
first `yield`, producing a fixed list `sorted_subgraphs` of `(inputs,
outputs)` snapshots taken against the *original* fgraph. Subsequent
mutations from `replace_all_validate` are not folded back into that
list.

The docstring claims the list is in "reverse topological order so they
can be safely replaced one at a time," and that ordering is enforced
this way:

```python
if not (unfuseable_ancestors_bitset & all_subgraphs_bitset):
    sorted_subgraphs.append(...)
else:
    # find an insertion position by walking previously-found subgraphs
    # in reverse and excluding their bitsets until the ancestor overlap
    # is gone
    ...
    sorted_subgraphs.insert(-(index + 1), ...)
```

The ordering criterion is `unfuseable_ancestors_bitset` - only
*upstream* dependencies. There is no symmetric handling for
*downstream* dependencies. When a new subgraph N is added, the planner
asks "does N depend on anything I have already?" and never asks "does
anything I have already depend on N?".

## Concrete failure on the reproducer

Instrumenting `apply` with the original (pre-fusion) toposort index of
each node yields the following plan for `min_repro.py`:

| yield# | starting node                                                   | nodes | outs                                                                                          | placement                            |
|--------|------------------------------------------------------------------|------:|------------------------------------------------------------------------------------------------|--------------------------------------|
| 1      | `Sub(Mul.0, Subtensor{:stop}.0)`                                 |     2 | `Sub.0`                                                                                        | discovered 3rd, inserted before D2   |
| 2      | `Mul(Add.0, Exp.0)`                                              |    10 | `Mul.0`                                                                                        | discovered 4th, inserted before D1   |
| 3      | `Switch(Isinf.0, [0.], Mul.0)`                                   |     2 | `Switch.0`                                                                                     | discovered 7th, inserted before D6   |
| 4      | `Add(Mul.0, Mul.0, ExpandDims.0)`                                |     2 | `Add.0`                                                                                        | discovered 8th, inserted before D6   |
| 5      | `Switch(Isinf.0, [0.], True_div.0)`                              |     3 | `Switch.0`                                                                                     | discovered 9th, inserted before D6   |
| 6      | `Switch(Isinf.0, True_div.0, [0.])` (multi-out)                  |    14 | 7 outputs incl. `Sigmoid.0` (`Sigmoid(Mul.0)`, original idx **28**)                            | discovered 6th, inserted before D1   |
| 7      | `Add(-1.8378…, Mul, Mul, Sum, Neg)` = `__logp`                   |    12 | `__logp`, `Mul.0`                                                                              | discovered 1st, appended             |
| **8**  | `Add(Mul.0, Mul.0, ExpandDims.0)`                                |   **5** | `Add.0` - inputs include the same `Sigmoid.0` (id matches yield 6's output)                  | discovered 5th, inserted before D2   |

Yield 6 is the upstream producer of yield 8. Its outputs include
`Sigmoid.0` (the output of `Sigmoid(Mul.0)`, originally at toposort
index 28), and yield 8's `inputs` contain that exact variable object.

When yield 6 is processed first, `replace_all_validate` rewires every
client of `Sigmoid.0` to the new `Composite(...)` node. The
`Variable` object for `Sigmoid.0` still exists, but
`fgraph.clients[Sigmoid.0]` is now empty.

When yield 8 is then attempted,
`toposort(yield8.outputs, blockers=yield8.inputs)` walks backwards
through the *current* fgraph. `Sigmoid.0` is dead - no node in the
live graph has it as an input - so the walk doesn't actually stop
there. Instead it crosses into the multi-output Composite that
replaced yield 6, and from there into *its* inputs
(`Subtensor{start:}(joined_inputs, 1)`, `Max{axes=None}`,
`Sum{axes=None}`, the multi-output Composite produced by yield 7,
etc.). The toposort that should have visited yield 8's 5 nodes instead
visits 36 nodes - many of them `Subtensor`, `Sum`, `Max`,
`DimShuffle` - and `replacement[inp]` is missing the first time
`inp` is one of those alien upstreams (`joined_inputs` in the captured
run; it would be whichever variable happens to be reached first).

So the trigger is precisely:

1. Yield 6 is a *producer* of yield 8 (one of its outputs is one of
   yield 8's inputs).
2. The planner inserted yield 6 at position 2 in `sorted_subgraphs`
   and yield 8 at position 7 (yield 6 before yield 8).
3. `replace_all_validate` for yield 6 invalidates yield 8's
   `inputs`-as-blockers.
4. `toposort` happily detours through the new graph, surfacing the
   real input vector (`joined_inputs`) as an unblocked leaf, and
   `elemwise_to_scalar` blows up.

Why did the planner choose this order? Yield 6's
`unfuseable_ancestors_bitset` covered nodes in yield 7's pre-existing
subgraph (the `Mul(200, Sub)` chain at original indices 13-17), so the
insert-loop walked backwards past D2/D5/D1, broke when those were
excluded, and placed yield 6 at insertion offset `-3`. There is no
loop iteration that asks whether yield 8 (already in
`sorted_subgraphs`) needs to come *before* yield 6.

## Why it is benign

`SequentialGraphRewriter.apply` wraps each sub-rewriter in a try /
except:

```python
# pytensor/graph/rewriting/basic.py
try:
    sub_prof = rewriter.apply(fgraph)
except AssertionError:
    raise
except Exception as e:
    if self.failure_callback:
        self.failure_callback(e, self, rewriter)   # logs + continues
        continue
    else:
        raise
```

The default callback (`SequentialGraphRewriter.warn`) `_logger.error`s
the rewriter name, "Traceback:", and the traceback, then returns
(unless `config.on_opt_error == "raise"`). So the partially-completed
fusion is abandoned, the fgraph remains in the state produced by the
already-applied yields, and compilation continues. With the bug
suppressed (see verification below) the posterior means and std
deviations match the buggy run to ~12 decimal places.

## Why I couldn't shrink the reproducer further

The reproducer in `min_repro.py` is the smallest one I found, and it
still requires pymc at runtime. I spent significant effort trying to
build a pure-pytensor reproducer that exercises the same planner path
and could not. Approaches tried:

1. Building the joined-input + logp + grad graph by hand with
   pytensor primitives (slice, reshape, expand_dims, logsumexp,
   softplus, normal-logp components, IncSubtensor or Join for the
   gradient).
2. Using pymc's own `rewrite_pregrad`, `CheckParameterValue`,
   `join_nonshared_inputs`, and `pymc.pytensorf.compile` (which
   applies `local_check_parameter_to_ninf_switch`) on a hand-built
   graph.
3. Capturing the exact pre-rewrite graph that pymc passes to
   `pytensor.function` (pickled it) and reproducing structural details
   - the `Subtensor{start:stop} → Reshape{0} → ExpandDims{axis=0}`
   scalar/vector roundtrip, the
   `Add(ExpandDims(Max), Log(ExpandDims(Sum(Switch(Isinf, Exp(MaxV),
   Exp(Sub(_, MaxV)))))))` form of `floor`, the
   `Add(-0.91893853, Mul(-0.5, Pow(Sub(obs, xi), 2)))` form of the
   Normal logp, the `Mul(-0.5, Sub(b, floor))` vs `Mul(0.5, Sub(floor, b))`
   inside the softplus argument of the Potential.

None of these reproductions triggered the planner mistake, even when
the pre-rewrite graphs looked structurally indistinguishable. The
*captured* pickle from `pm.sample` triggers deterministically when
fed straight to bare `pytensor.function` (no pymc imported), so the
bug is squarely in pytensor's fusion planner - pymc just happens to
construct a graph that exercises it. I gave up after ~30 minimization
attempts; given more time I would (a) instrument
`find_fuseable_subgraphs` to dump the per-call planner state in both
the triggering and non-triggering graphs and diff them to find the
exact structural feature, and (b) try shrinking the captured pickle
itself by removing nodes incrementally. Direction (b) is mechanical
but tedious enough that I judged the cleaned-up pymc reproducer
sufficient to file the issue.

## Suggested fix

Two viable directions:

1. **Validate at the call site (small, localised).** Before calling
   `self.elemwise_to_scalar(inputs, outputs)` in
   `FusionOptimizer.apply`, check that every variable in `inputs`
   still has clients in the current fgraph. If any is dead, skip the
   subgraph - its planned role as a blocker has been invalidated by
   an earlier fusion in this same pass. The next `apply` invocation
   will re-plan against the updated graph. Verified on the
   reproducer:

   ```python
   for inputs, outputs in find_fuseable_subgraphs(fgraph):
       if (len(inputs) + len(outputs)) > max_operands:
           ...
           continue

       _fg_clients = fgraph.clients
       if any(not _fg_clients.get(inp) for inp in inputs):
           # stale frontier - a previous fusion replaced one of these
           # inputs and they no longer block the toposort
           continue

       scalar_inputs, scalar_outputs = self.elemwise_to_scalar(inputs, outputs)
       ...
   ```

   Turns 3 `ERROR` records into 0 with no change in `a`/`b` posterior
   means or stds.

2. **Fix the ordering (deeper).** When a new subgraph N is inserted
   into `sorted_subgraphs`, also walk *forward* through the existing
   list to ensure no already-present subgraph M has any of N's outputs
   as one of M's inputs. If it does, N must come *after* M (so M, the
   downstream consumer, is replaced first). The
   `unfuseable_clients_bitset` already computed during N's exploration
   is exactly the right input - it's currently discarded once the
   subgraph is closed. This removes the root cause; (1) is a
   defensive backstop.

Either change is small. (2) is the right long-term fix; (1) is a
one-line guard that also catches any other stale-frontier corner case.

## Verification

Numbers from running the original 5-prior reproducer (the noisy one
I started from, before minimization), upgraded to
`draws=50, tune=50, chains=2`:

| build      | `b0` mean         | `xi_u` mean        | `b0` std          | `xi_u` std        | ERROR records |
|------------|-------------------|--------------------|-------------------|-------------------|--------------:|
| unpatched  | -0.227704492873…  | 0.908918354363…    | 0.995090493053…   | 0.558715785987…   |             3 |
| patched(1) | -0.227704492873…  | 0.908918354363…    | 0.995090493053…   | 0.558715785987…   |             0 |

Identical to roughly 12 decimal places - confirming the rewrite that
gets skipped really was redundant cleanup that other rewriters cover.

## Cited code

- `pytensor/tensor/rewriting/elemwise.py:526-928` - `FusionOptimizer`
  (the entire class; the planner bug is in the `apply` body's
  `find_fuseable_subgraphs` insertion logic, lines 836-866 in 3.0.2).
- `pytensor/tensor/rewriting/elemwise.py:533-551` -
  `elemwise_to_scalar` (the crash site, line 538).
- `pytensor/graph/rewriting/basic.py:285-303` -
  `SequentialGraphRewriter.apply`'s try/except that swallows the
  `KeyError`.
- `pytensor/graph/rewriting/basic.py:227-239` -
  `SequentialGraphRewriter.warn` (the default `failure_callback` that
  logs `_logger.error` and returns).
- `pymc/pytensorf.py:573-599` - `join_nonshared_inputs` (creates the
  `joined_inputs` variable whose name shows up in the traceback).
