# `stable_smooth` follow-up work (agent handoff)

This document is the self-contained briefing for a cloud agent picking
up the remaining work on `dev/taylor/stable_smooth`.  Read it end-to-end
before touching code.  It assumes nothing about the conversation that
produced the current state -- only what's in the repo at HEAD of branch
`dev-taylor-remainder`.

## 0. Orientation

### Repo & environment

- **Working tree:** `/home/mares/repos/pytensor`
- **Branch:** `dev-taylor-remainder` (do *not* PR to `main` without
  human review)
- **Source:** `dev/taylor/taylor_remainder.py`
- **Tests:** `dev/taylor/test_taylor_remainder.py`
- **Design notes:** `dev/taylor/stable_smooth_design.md`
- **Python env:** `micromamba run -n pytensor-dev python ...`
  (the repo's `.venv/bin/python` is missing pytensor; do not use it,
  do not try to `uv run` -- it builds from source and fails on `cc`.)
- **Run tests:**
  ```bash
  cd /home/mares/repos/pytensor/dev/taylor
  micromamba run -n pytensor-dev python -m pytest test_taylor_remainder.py
  ```
  Full suite currently 56 passing in ~11s.

### Commit & PR discipline (read before touching git)

- Atomic, frequently committed; one logical change per commit.
- Pre-commit runs `ruff format` -- if it reformats, re-stage and commit
  again.  Hook output `1 file reformatted` is normal, not a failure.
- **Never** skip hooks (no `--no-verify`).
- **Never** bundle lock-file changes with code/manifest changes -- lock
  files must be in their own commit.
- Commit message style: terse subject, blank line, body explaining
  *why*.  Look at `git log dev/taylor/` for recent examples.
- Do not commit `.env`, credentials, or large binaries.  If you create
  scratch scripts under `/tmp/` they're fine to leave there.

### What `stable_smooth` is

```python
stable_smooth(numerator, x, a, *, denominator_degree, cancellation_order=0,
              dtype=None, inline=False)
```

User-facing wrapper around `taylor_remainder` that:

1. Auto-picks `order` and `eps` via `_min_order_and_eps` based on the
   numerator's Taylor coefficients at `a` and the user-declared
   `cancellation_order` (rel_err contract on the numerator's
   evaluation: `rel_err ≤ ε_m · |v|^{-c}`).
2. Wraps the forward graph in an `OpFromGraph` whose `pullback` returns
   another `stable_smooth` call, so `pt.grad` is closed under
   `stable_smooth`.  Identity used:
   `R_n'(f)(x) = R_1[R_{n-1}(f') − n·R_n(f)](x)`.
3. Passes child caches analytically via the `_coefficients=` pipeline
   (the bracket's `j`-th Taylor coefficient at `a` is `j · c_{j+n}^f`
   independent of `n`), so the child's cache never re-traverses
   `pt.grad` through the parent OpFromGraph.

Forward usage:

```python
import pytensor.tensor as pt
from taylor_remainder import stable_smooth

x = pt.dscalar("x")
sinc = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
# sinc evaluates to sin(x)/x stably across x=0.

import pytensor
fn = pytensor.function([x], sinc)
fn(0.0)   # -> 1.0
fn(0.5)   # -> sin(0.5)/0.5 to machine precision
```

Read `dev/taylor/stable_smooth_design.md` for the full design spec
(user story, four pitfalls, error model, identity derivation).

## 1. Open follow-ups

Three open tasks remain.  Each section below is self-contained: problem
statement → what's been learned → candidate approaches → acceptance
criteria → where in the code to start → testing strategy.

Order them however makes sense given your investigation; the dependency
graph is essentially flat (none of the three blocks the others).

### Status

- **Task A (perf cliff):** done, via an upstream pytensor fix +
  default-flip on our side.  The cluster of OFG-cloning commits
  (`6458acc`, `b39fced`, `a11a9b1`, `a821179`, `7821c7e`) that landed
  shortly after `rel-3.0.0` reduced inline-time cloning to the point
  where `inline=True` is the strictly faster path at every depth
  measured.  The `stable_smooth` default flipped from `inline=False`
  to `inline=True`; depth-5 sinc now goes from build to second eval in
  ~10 s total (was >2 min on the same configuration before the
  upstream fixes).  Locked in by
  `test_stable_smooth_depth5_under_wallclock_budget` with a 30 s budget.
- **Task B (vector input):** done.
  `stable_smooth` accepts elementwise vector `x`.  The earlier
  scalar-only `NotImplementedError` test
  (`test_stable_smooth_vector_input_raises_helpful_error`) was flipped
  into a positive forward test, and seven more vector tests cover the
  grad chain, n=2, nonzero-`a`, float32, depth-2 grads, and documented
  non-elementwise misuse.  The plan's `clone_replace(numerator, {x:
  x_s})` step doesn't actually work in pytensor (the type filter
  rejects vector→scalar substitution), so the implementation uses a
  custom `_scalarize_elementwise` graph walker + `pt.squeeze` to
  produce the scalar surrogate.  `cache=` cross-call sharing is still
  scalar-only -- locked in by `test_stable_smooth_vector_input_cache_raises`
  -- because a vector user can't easily produce a scalar-surrogate
  cache that the validator would accept.
- **Task C (cross-call cache):** done.
  `stable_smooth(..., cache=TaylorAtPoint(...))` shares a single
  coefficient chain across `denominator_degree`s.  Validator rejects
  `(x, a, numerator)` mismatches: identity check on `x`, value check
  on `a`, structural equality (`equal_computations`) on the numerator
  so users can naturally rebuild the same expression without threading
  the same TensorVariable through every call.

The Task A, B, and C sections below are kept for historical context --
they're the briefing that the prior agent ran against and capture the
reasoning that landed each fix.

---

## Task A.  Fix the depth-5 first-eval cliff (dedup op cascade)

**Symptom.**  After taking `pt.grad` enough times that the chain depth
exceeds about 4, the very first call to the compiled function takes
tens of seconds.  `inline=True` (already shipped as a user-facing
toggle) sidesteps this by paying the cost at `pytensor.function` time
instead, but neither path is fast.  Concrete numbers at depth 5:

| Phase                             | `inline=False` (default) | `inline=True` |
| --------------------------------- | -----------------------: | ------------: |
| `stable_smooth` construction      | ~50 ms                  | ~50 ms       |
| 5×`pt.grad`                       | ~150 ms (linear)        | ~150 ms      |
| `pytensor.function([x], cur)`     | ~1.5 s                  | **~47 s**    |
| `fn(0.0)` first call              | **~34 s**               | ~0 s          |
| `fn(0.0)` subsequent              | ~0.1 s                   | ~0 s          |

The total wall-clock is similar (~35 s either way) but the *kind* of
cost differs.

**Root cause (confirmed, this is the part that's done).**

At depth 5 we construct only 21 unique `OpFromGraph` instances (linear
growth, see `trace_ops.py` reproducer below).  But by the time eval
runs, the graph contains ~1000 distinct OpFromGraph instances calling
`.perform()`.  The extras are *clones* introduced by pytensor's graph
rewriter (canonicalize, etc.) during `pytensor.function` build.

The clone is a *shallow copy*: it gets a fresh `fgraph` (`res.fgraph =
res.fgraph.clone(clone_inner_graphs=True)`) but the `_fn` attribute
(the cached compiled inner function, line ~915 of
`pytensor/compile/builders.py`) is reset to `None` on the clone.  So
every clone re-compiles its inner function from scratch when first
called -- O(1000) lazy `pytensor.function(...)` invocations, each
~30 ms in `FAST_COMPILE` mode.

Where to look in the code:

- `pytensor/compile/builders.py:936` -- `OpFromGraph.clone`
- `pytensor/compile/builders.py:915` -- the `fn` property that lazily
  compiles
- `pytensor/compile/builders.py:399` -- `self._lop_op_cache = {}` per
  instance (also not shared with clones)

**What we've already tried (don't redo these).**

1. `inline=True`.  Shipped as a user-facing option; equally slow in
   total, just front-loaded.
2. Reducing op count at construction.  Already linear; not the
   bottleneck.
3. Avoiding `pt.grad` through embedded `op(...)` apply nodes in the
   pullback by reconstructing R symbolically.  That blew the graph up
   exponentially (the path we abandoned in commit 13e135f8e in favor
   of the orphan-passthrough approach).

**Promising avenues for further work.**

The fix lives in pytensor itself, not in `stable_smooth`.  Two distinct
shapes:

1. **Share `_fn` across structurally-equivalent clones.**  Add a
   class-level (or module-level) cache in `OpFromGraph`, keyed by some
   identity of the inner graph (e.g., a structural hash of
   `(inner_inputs, inner_outputs)` after canonicalization).  Clones
   look up the cache before compiling.  Hardest part: defining the hash
   so it survives the rewriter's tree edits.  Look at
   `pytensor.graph.basic.equal_computations` for a starting point.

2. **Make `OpFromGraph.clone` preserve `_fn`.**  If `clone()` is purely
   a shallow `copy(self)` modulo `fgraph.clone(clone_inner_graphs=True)`,
   then it could keep `self._fn` *if and only if* the new
   `inner_inputs`/`inner_outputs` are positionally compatible with the
   compiled function (they are, since `function()` returns a callable
   indexed by position, not by Variable identity).  Test by
   monkey-patching `clone` in `count_ops.py` (see "Reproducers") and
   measuring the eval cliff.

3. **Make the clone unnecessary.**  Investigate which rewriter pass is
   cloning OFGs and why.  If it's `canonicalize` doing a no-op
   structural rebuild, maybe an early-out is possible.  Use
   `pytensor.config.optimizer="None"` to confirm: if the cliff
   disappears, you've confirmed it's optimization-driven, not
   evaluation-driven.

A non-pytensor fallback would be **eager construction up to a
user-declared `max_depth`**: build `op_0 ... op_K` upfront sharing
references; pt.grad would still clone, but at least construction
wouldn't lazily fire ops we'll never use.  This won't fix the cliff,
just bound it.

**Acceptance criteria.**

- For sinc at depth 5: `pytensor.function([x], cur)` followed by
  `fn(0.0)` should complete in < 5 s total.
- For sinc at depth 8: should complete in < 30 s total.
- 56 existing tests still pass.
- Add a new test that asserts the depth-5 wall-clock budget (using
  `time.perf_counter`) so the regression doesn't silently come back.

**Reproducers.**

`/tmp/probe_depth/trace_ops.py` and `/tmp/probe_depth/count_ops.py`
were left in place by the previous session.  If they've been cleaned
up, here's `trace_ops.py` in full:

```python
import collections, sys
sys.path.append("/home/mares/repos/pytensor/dev/taylor")
import pytensor
import pytensor.tensor as pt
pytensor.config.mode = "FAST_COMPILE"
from pytensor.compile.builders import OpFromGraph
from taylor_remainder import stable_smooth

op_count = [0]
_orig_init = OpFromGraph.__init__
def _counting_init(self, *args, **kwargs):
    op_count[0] += 1
    _orig_init(self, *args, **kwargs)
OpFromGraph.__init__ = _counting_init

x = pt.dscalar("x")
cur = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
print(f"after build: {op_count[0]} ops")
for k in range(5):
    cur = pt.grad(cur, x)
    print(f"after grad #{k+1}: {op_count[0]} ops")

import time
t0 = time.time()
fn = pytensor.function([x], cur)
print(f"function() {time.time()-t0:.2f}s ({op_count[0]} ops cumulative)")
t0 = time.time()
v = float(fn(0.0))
print(f"first eval: {v} ({time.time()-t0:.2f}s)")
t0 = time.time()
v = float(fn(0.0))
print(f"second eval: {v} ({time.time()-t0:.2f}s)")
```

Run from outside the repo root (e.g. `cd /tmp/probe_depth && micromamba
run -n pytensor-dev python trace_ops.py`) so `pytensor` resolves to the
installed package rather than the `pytensor/` source dir.

---

## Task B.  Vector/elementwise input support

**Symptom.**  `stable_smooth` currently raises
`NotImplementedError("stable_smooth currently supports scalar 'x'
only ...")` for any non-scalar input.  Test:
`test_stable_smooth_vector_input_raises_helpful_error` locks this in.

**Why it's hard (specific failure modes).**

Two distinct places assume scalar `x`:

1. `TaylorAtPoint.value_at_a(m)` (line 276 of `taylor_remainder.py`):

   ```python
   v = clone_replace(d, {self.x: self._a_const})
   ```

   Here `self._a_const = pt.constant(a, dtype=x.dtype)` is a scalar.
   `clone_replace` substitutes `self.x` (vector) with a scalar
   `_a_const`, which fails the type-filter check in
   `pytensor/tensor/type.py:294` with the message *"Cannot convert
   Type Scalar(float64, shape=()) into Type Vector(float64,
   shape=(?,))"*.

2. `stable_smooth.pullback` (line ~1100 of `taylor_remainder.py`):

   ```python
   f_prime_at_xi = pt.grad(num_at_xi, xi)
   ```

   `pt.grad` requires a *scalar* cost.  For vector `num_at_xi`
   (elementwise application of `f` to vector `xi`), this fails
   immediately.

**The plan (untried).**

1. Introduce a **scalar surrogate** for the cache.  In `stable_smooth`:

   ```python
   x_s = pt.scalar(dtype=x.dtype)  # scalar surrogate
   numerator_s = clone_replace(numerator, {x: x_s})
   cache = TaylorAtPoint(numerator_s, x_s, a, coefficients=_coefficients)
   ```

   The cache's numeric values are scalar floats wrapped as
   `pt.constant`s -- they broadcast cleanly against vector `inner_x` in
   the forward switch (the polynomial branch is Horner over scalar
   coefficients times vector `t`; the closed branch is `(vector_f -
   scalar_P) / vector_t**n`, all elementwise).

2. **Replace `pt.grad(num_at_xi, xi)`** in the pullback with the
   elementwise-derivative form for elementwise `f`:

   ```python
   f_prime_at_xi = pt.grad(num_at_xi.sum(), xi)
   ```

   For elementwise `f`, the Jacobian is diagonal and `pt.grad(sum(f),
   x)` recovers `f'(x)` elementwise.  Document and assert this
   restriction up front: if the user passes a non-elementwise
   numerator, behavior is undefined.  (You may want to add a
   diagonality check via `pytensor.gradient.jacobian` or just gate on
   *the user passing a flag* `elementwise=True`.)

3. **Tangle with `_orphan_substitutions`.**  The orphan `R_orphan` is
   vector (it's the parent op's output, which is the same shape as
   `x`).  When the pullback hands the recursive child stable_smooth
   `_orphan_substitutions={R_orphan: op(xi)}`, `op(xi)` is a vector
   call -- but the recursive child then wants a scalar surrogate for
   *its* cache.  Specifically:

   ```python
   # in recursive stable_smooth
   x_s_child = pt.scalar(...)
   numerator_s_child = clone_replace(bracket, {x: x_s_child})
   ```

   `bracket` contains the orphan `R_orphan` (vector).  Substituting `x
   → x_s_child` (scalar) inside `bracket` will try to feed scalar where
   `R_orphan` is connected as vector -> type error.

   **Solution sketch:** apply `_orphan_substitutions` first using a
   *scalar* version of `op` (i.e., a clone of `op` with scalar inputs),
   then substitute `x → x_s_child`.  This requires keeping the
   "scalar variant" of `op` alongside the vector one in the closure.
   Concretely: when building `op`, also build `op_scalar` with
   `inputs=[x_s]` and `outputs=[inner_remainder_scalar]`, and stash
   `op_scalar` in the pullback's closure.  Then
   `_orphan_substitutions = {R_orphan: op_scalar(x_s_child)}` for the
   recursive call's *cache*, while the actual vector graph still uses
   `op(xi)`.

   This is the major design wrinkle.  Sketch it carefully on paper
   before coding.

**Acceptance criteria.**

- Vector input case from `test_stable_smooth_vector_input_raises_helpful_error`
  flips: instead of raising, it should compute sinc elementwise
  correctly, including at array entries that are exactly zero.  Test
  body:

  ```python
  x = pt.dvector("x")
  sinc = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
  fn = pytensor.function([x], sinc)
  out = fn(np.array([0.0, 1e-10, 1e-4, 0.5, 1.0, -0.3]))
  expected = np.array([1.0, 1.0, 1.0 - 1e-8/6,
                       math.sin(0.5)/0.5, math.sin(1.0),
                       math.sin(-0.3)/-0.3])
  np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)
  ```

- Grad chain for vector inputs: `pt.grad(sinc.sum(), x)` should give
  sinc' elementwise (compared to mpmath reference per element).
- All existing 56 scalar tests still pass.
- New test: non-elementwise numerator (e.g.,
  `numerator = pt.sum(pt.sin(x)) * pt.ones_like(x)`) either raises an
  informative error or is explicitly documented as
  unsupported-behavior.

**Where to start.**

- `dev/taylor/taylor_remainder.py:200-310` -- `TaylorAtPoint`
  (`deriv`, `value_at_a`).  Don't modify `TaylorAtPoint` itself; use a
  scalar surrogate from the `stable_smooth` side.
- `dev/taylor/taylor_remainder.py:1050-1140` -- `stable_smooth` body
  and its pullback.

---

## Task C.  Cross-call `TaylorAtPoint` memoization

**Status: low-priority polish.**  Within a single `stable_smooth` call,
parent → child coefficient sharing already works via the
`_coefficients` pipeline (the child's cache pulls from a generator
that consumes from the parent's cache; numerics computed once per
index).  The remaining case is *separate user calls* to
`stable_smooth` (or `taylor_remainder`) that share the same `(f, x,
a, dtype)`, e.g.:

```python
sinc_n1 = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
sinc_n2 = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=2)
```

Each call constructs its own `TaylorAtPoint(pt.sin(x), x, 0.0)`, which
runs the `pt.grad` chain redundantly.

**The plan (low-effort).**

Add an optional `cache` parameter to `stable_smooth` that the user
constructs once and shares:

```python
cache = TaylorAtPoint(pt.sin(x), x, 0.0)
sinc_n1 = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1, cache=cache)
sinc_n2 = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=2, cache=cache)
```

This mirrors `taylor_remainder`'s existing `cache=` parameter (see
`taylor_remainder` signature, line ~840).  No magic memoization needed.

**Don't do (rejected approaches).**

- Module-level dict keyed by `id(f)`.  Brittle: `id` is reused as
  Python objects are garbage-collected, and users who rebuild the same
  expression would not benefit.  WeakValueDictionary would help with
  the GC issue but still doesn't unify structurally-equal expressions.
- Hashing the expression's graph structure.  Overkill for the actual
  user benefit; the user can pass the cache explicitly when they care.

**Acceptance criteria.**

- New optional `cache: TaylorAtPoint | None = None` kwarg on
  `stable_smooth`.  When passed, used instead of constructing a fresh
  one.
- Validate that the supplied cache matches `(numerator, x, a)`
  (compare `cache.x is x`, `cache.a == a`; for the numerator, compare
  `cache.f is numerator` or similar weak structural check).  Raise
  clear error on mismatch.
- Add a test that constructs the cache once, calls `stable_smooth`
  twice with different `denominator_degree`, and verifies (via a
  counter on `TaylorAtPoint.deriv` or `pt.grad` call count) that
  `pt.grad` is invoked fewer times than two independent calls would.

---

## 2. Cross-cutting

### Test style

- Use `mpmath` at 50 dps (`mp.mp.dps = 50` is set globally in
  `test_taylor_remainder.py:27`) for reference values when the
  *float-domain* reference would itself suffer cancellation (e.g.,
  small-`x` evaluations).  See
  `test_stable_smooth_n2_cancellation_order_2_matches_sinc_prime` for
  the pattern.
- Tests use `pytensor.config.mode = "FAST_COMPILE"` (set at the top of
  the file) to keep compile times manageable.
- Tolerance pattern: `math.isclose(got, expected, rel_tol=1e-12,
  abs_tol=1e-15)` for typical, looser (`rel_tol=1e-10`) for cases
  through the chain or with cancellation.
- Use `pytest.raises` for failure modes -- see
  `test_stable_smooth_negative_n_raises`.

### File-level conventions

- `taylor_remainder.py` mixes core (`TaylorAtPoint`,
  `taylor_remainder`) with the user-facing wrapper (`stable_smooth`).
  Don't split into multiple files without a good reason -- the design
  doc and the implementation co-evolve.
- Comments: lead with *why*, not *what*.  Look at the pullback (line
  ~1080-1140) for the style.
- Don't add `# removed for X` or `# old code below` style markers --
  just delete the old code; the commit log is the history.

### When in doubt

- Read `dev/taylor/stable_smooth_design.md` end-to-end.  It captures
  the user story, the four numerical pitfalls being addressed, the
  identity used in the grad chain, and the error-model formulas.
- The recent commit log on `dev/taylor/` (`git log --oneline dev/taylor/`)
  is the second-best context.
- Don't refactor for the sake of it.  Three similar lines is better
  than a premature abstraction.

### Out of scope

- Performance work on `taylor_remainder` itself (not `stable_smooth`).
  The lower-level routine is solid; speedups should target
  `stable_smooth`'s grad chain or pytensor's clone semantics.
- API changes that would break the existing 56 tests without a clear
  reason in the design doc.
- Anything outside `dev/taylor/` (the rest of the pytensor codebase is
  not your battlefield for this work).
