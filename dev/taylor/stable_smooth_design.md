# `stable_smooth` — user-facing wrapper for `taylor_remainder`

Design notes for the planned `stable_smooth(...)` API. Not yet implemented;
this document is the spec we're working from.

## User story

The user wants to extend PyTensor with smooth-but-singular functions like
`sinc(x) = sin(x)/x`, `expm1(x)/x`, `log1p(x)/x`, `(cos(x)−1)/x²`,
`(x·cos(x)−sin(x))/x²`, etc. They're happy to write down the *structure*
(numerator, expansion point, denominator order). What they refuse to think
about is the numerical stability of every derivative.

The pitfalls being solved, in increasing order of insidiousness:

1. **Naive `pt.switch(x==0, c0, naive(x))`.** Forward eval is usually fine
   (Sterbenz saves things like `sin(x)/x` for moderately small `x`).
   `pt.grad` chains through the naive branch, which has catastrophic
   cancellation in a neighborhood `[−δ, δ] \ {0}` that the literal
   `x == 0` test doesn't cover.

2. **`pt.switch(x==0, 1, expm1(x)/x)`.** The naive branch is fine even for
   tiny `x`. But `pt.grad(switch(cond, 1, naive)) = switch(cond, 0,
   grad(naive))` — the *constant* branch has zero gradient, so the
   derivative at `x = 0` is wrong (true value: `1/2`; this gives `0`).

3. **Picking `eps` wrong for dtype.** A hardcoded `eps = 1e-7` is wrong for
   float32, float16, and wrong after `k` derivatives shrink the
   polynomial-branch comfort zone.

4. **Multi-term cancellation.** Different functions have different
   cancellation depths. `sin(x)/x` has one near-zero subtraction
   (Sterbenz-safe); `(cos(x) − 1)/x²` has multi-term cancellation that
   needs a wider polynomial window.

## API

```python
stable_smooth(
    numerator,                # PyTensor expression for f(x)
    x,                        # the variable (scalar or vector)
    a,                        # expansion point
    *,
    denominator_degree,       # n in f / (x - a)^n
    cancellation_order=0,     # the user's evaluation-precision contract; see below
    cache=None,               # optional pre-built TaylorAtPoint for cross-call sharing
)
```

Knobs the API deliberately does *not* expose:

- `order` (polynomial-branch length) and `eps` (switch threshold) are
  picked by `_min_order_and_eps` from the numerator's coefficient
  sizes at `a`; the user shouldn't have to think about either.
- `dtype` is read from `x.dtype`; no override is needed since the
  auto-eps formula is scale-invariant in `f` and the only dtype
  question is "what is `eps_machine`?".
- `inline` (OpFromGraph inlining strategy) is always `True`.  The
  upstream OFG-cloning fixes after `rel-3.0.0` made it strictly the
  faster path; the lazy-per-OFG path no longer has a use case.
- `cancellation_order` is the one remaining unresolved API wart: it
  encodes a precision contract the graph can't reliably infer, so we
  keep it as an explicit kwarg defaulted to 0.  Future work could
  split it out into a separate "I know my numerator suffers
  cancellation" entry point.

| Parameter | Role |
|---|---|
| `numerator` | PyTensor expression `f(x)`. |
| `x`, `a` | Variable and expansion point. |
| `denominator_degree` (n) | Builds `R_n(f) = (f − P_{n−1})/(x−a)^n`. User's free choice — different `n`s give different smooth functions. |
| `cancellation_order` (c) | User's contract on `f`'s evaluation precision: `rel_err(computed f(v)) ≤ ε_m · |v|^{−c}`. Default 0 = pristine. **Independent of `k_lead`** (first-nonvanishing Taylor index): e.g. `exp(−1/x²)` has `k_lead = ∞` but `c = 0` (pristine evaluation). A user feeding `x·cos(x) − sin(x)` declares `c = 2` because the literal expression loses 2 orders of magnitude in `v` during the subtraction. |

The signature is closed under `pt.grad`: each `pt.grad(stable_smooth(...)(x),
x)` returns another `stable_smooth(...)` call. The framework tracks
`derivative_depth` internally (not user-set).

## Two-branch identity for the grad chain

```
R_n'(f)(x)  =  R_1[ R_{n−1}(f') − n·R_n(f) ](x)
```

(For `n = 1` the inner `R_{n−1}(f') = R_0(f') = f'`, so this collapses
to the simpler `R_1'(f) = R_1[f' − R_1(f)]`. An earlier draft of this
doc wrote `f'` in place of `R_{n−1}(f')` everywhere -- correct at
`n = 1` only.)

The bracketed quantity vanishes at `a` (both terms equal
`f^(n)(a)/(n−1)!` there) so the L'Hôpital-style `R_1` limit is
well-defined. Crucially, `denominator_degree` collapses to `1` after the
first derivative and stays at `1`; only the numerator gets symbolically
more complex as the chain deepens. `cancellation_order` of the next-level
expression follows from the precision contract of the new numerator,
which is built from the previous level's `stable_smooth` plus elementary
operations on `f`'s derivatives.

### Implementation status

- General `n ≥ 1` works.  For `n > 1` the pullback recurses with
  `stable_smooth(f', x, a, denominator_degree=n−1)` to express
  `R_{n−1}(f')`; the bracket's `j·c_{j+n}^f` coefficient formula is
  `n`-independent so it carries through unchanged.
- Grad chain works correctly through at least depth 5 in float64 and
  stays under a ~30 s wall-clock budget (locked in by
  `test_stable_smooth_depth5_under_wallclock_budget`).  The earlier
  "compile time grows ~6× per level beyond k=4" symptom was the
  inline=False lazy-compile cascade; flipping the default to
  inline=True (after the upstream OFG-cloning fixes) made the cost
  effectively per-graph rather than per-clone, so deeper depths are
  bounded by the inliner's work rather than the per-OFG `_fn` rebuild.
  See `Performance` below for the measured numbers.

## General `f / g` via composition

If both vanish at `a` to order ≥ `k` for some `k ≥ 1`:

```python
stable_smooth(f, x, a, denominator_degree=k, ...) \
    / stable_smooth(g, x, a, denominator_degree=k, ...)
```

No dedicated ratio primitive needed.

## Multivariate singularities via derived-expression composition

`stable_smooth` builds a Taylor expansion in a single scalar/vector
variable.  Higher-arity functions with a singularity in one direction
can often be reduced to that univariate form by *factoring*: identify
a univariate `h(u)` carrying the singularity, build it with a fresh
leaf, then `clone_replace` the leaf with the user's derived expression
at the call site.

### Worked example: `(1 + ξ·z)^(−1/ξ)`

This is the GEV-style link from extreme value theory.  Domain is
`1 + ξ·z > 0` for all real `ξ`.  At `ξ = 0` the value is
`exp(−z)` (Gumbel limit) — `1^∞` indeterminate if evaluated directly.

Decomposition:

```
f(ξ, z)  =  exp( −log(1 + ξ·z) / ξ )
         =  exp( −z · log1p(ξ·z) / (ξ·z) )
         =  exp( −z · h(ξ·z) ),    where  h(u) = log1p(u) / u.
```

`h(u)` is a univariate smooth-but-singular function (the familiar
`log1p/x` case from this design's user story).  `stable_smooth` gives
a stable `h` directly:

```python
import pytensor.tensor as pt
from pytensor.graph.replace import clone_replace
from taylor_remainder import stable_smooth

xi = pt.dscalar("xi")
z  = pt.dscalar("z")

# Build h(u) = log1p(u)/u once over a leaf u.
u = pt.dscalar("u")
h_at_u = stable_smooth(pt.log1p(u), u, 0.0, denominator_degree=1)

# Substitute u → ξ·z.  clone_replace just retargets the OpFromGraph's
# call site; the OFG itself (and its pullback) is unchanged.
h_at_xz = clone_replace(h_at_u, {u: xi * z})

f = pt.exp(-z * h_at_xz)
```

Why this is enough.  `pt.grad(f, ξ)` and `pt.grad(f, z)` both go
through `h_at_u`'s OpFromGraph pullback.  The pullback returns the
derivative w.r.t. `h`'s inner input only; pytensor's outer chain rule
multiplies by `d(ξ·z)/dξ = z` (or `d(ξ·z)/dz = ξ`) to recover the
gradient w.r.t. ξ or z.  No specialized bivariate machinery needed.
The pullback at `ξ = 0` is correct because the inner gradient of
`h(u)` is itself a `stable_smooth` (closed under `pt.grad`), so the
limit at `u = 0` is computed via the polynomial branch -- not by
attempting `log1p/u` directly.

Vectorized variant: build `h` over a vector leaf and the substitution
broadcasts the same way.  For `ξ` a vector and `z` a scalar:

```python
xi_v = pt.dvector("xi")
z    = pt.dscalar("z")
u_v  = pt.dvector("u")
h_v  = stable_smooth(pt.log1p(u_v), u_v, 0.0, denominator_degree=1)
f_v  = pt.exp(-z * clone_replace(h_v, {u_v: xi_v * z}))
```

### When the pattern applies (and doesn't)

The composition reduces `g(ξ, z, ...)` to `g(u, z, ...) = ... h(u) ...`
with `u = derived_expression(ξ, z, ...)`.  Two requirements:

1. **`g` factors through `u`.**  All the "tricky" `ξ` (or `z`)
   dependence must enter `g` through `u`.  The "outer" arithmetic
   wrapping `h(u)` -- e.g. the `exp(−z · ...)` in the GEV case -- is
   free to depend on the other variables; it stays in the outer graph
   and inherits stability from `h(u)`'s polynomial branch.

   Counterexample: `h(ξ·z) + ξ²` doesn't factor through `u = ξ·z`
   alone -- the `ξ²` term keeps an independent `ξ` dependence that
   breaks the substitution trick.  For such hybrids, you'd build the
   stable part as above and add the residual outside.

2. **The constant term vanishes at `u = a`** (the usual
   `denominator_degree` precondition).  In the GEV case `log1p(u)`
   vanishes at `u = 0` so `denominator_degree=1` is well-posed for any
   `(ξ, z)`.  An expression like `h(ξ·z) = (ξ·z + ξ²) / (ξ·z)` would
   need a different decomposition -- the numerator only vanishes at
   `u = 0` when `ξ = 0`, not for arbitrary `(ξ, z)`.

### Why we don't extend `stable_smooth` to accept derived `x`

A tempting alternative is:

```python
stable_smooth(pt.log1p(xi * z), xi * z, 0.0, denominator_degree=1)
```

That would require `stable_smooth` to (a) detect that `xi * z` is
derived, (b) build an internal leaf surrogate, (c) substitute the
derived subgraph into the result.  Step (b) duplicates work that
`clone_replace` already does cleanly, and step (a) makes the API
fragile: the user would have to remember which subgraph they
considered "the expansion variable" when the numerator references it
multiple times.  The explicit-leaf pattern documented above makes the
factoring visible at the call site, which is also the place a future
reader has to understand to maintain the code.

## Order growth: minimum sufficient, lazy

**Anti-pattern (was tempted):** eager + generous, e.g.
`order = max(declared_depth + 8, 12)`. *Wrong because* gradients can be
very expensive — `pt.grad` on a non-trivial `f` (e.g. `cos(K · x²)`) is
O(seconds) per derivative once the chain rule expands. Padding wastes work
the user may never need.

**Right approach:** order grows lazily, minimum sufficient at each level:

```
order = derivative_depth + 1
loop:
    ensure cache.numeric_coeff(n + order − 1) is pulled    # extend polynomial
    find smallest m > n + order − 1 such that |cache.numeric_coeff(m)| > 0
        (pulls coefficients lazily as it searches; this is v_trunc)
    compute eps from the analytic formula using v_trunc
    if closed_branch_rel_err_bound(eps; c) ≤ tol_rel:
        done
    order += 1
```

Key invariants:

- We *always* pull at least one nonzero coefficient past the polynomial
  degree — it's the truncation-error estimator (`v_trunc`).
- For sparse series (parity gaps, etc.) the "find smallest `m > … nonzero`"
  step may skip zeros. The existing `max_extra = 4` heuristic can be
  replaced by "pull until nonzero".
- For explicit-coefficients mode with a finite list, exhaustion raises
  `IndexError` — the user under-provisioned for the requested derivative
  depth.

## Error model with `cancellation_order`

Closed-branch rel_err bound generalized:

```
rel_err_closed(v; c)
    ≤ ε_m · max(1, |v|^{−c}) · Σ_{i ∈ [0,n)} |c_i · v^i| / (|c_{k_lead}| · v^{k_lead})
    + ε_m                                                  # division-rounding floor
```

For pristine `f` (c = 0) and typical analytic case, `max(1, |v|^0) = 1`
and we recover the current `closed_branch_rel_err_bound`. For cancelled
`f` (c > 0) and small `v`, the `|v|^{−c}` factor amplifies the bound,
stretching `eps_lower` outward — `auto_eps` correspondingly picks a wider
polynomial window.

## Cache sharing

Memoize `TaylorAtPoint` by `(f-graph-identity, x, a, dtype)`. The grad
chain builds nested `stable_smooth` Op invocations; all levels share one
cache per `(f, x, a, dtype)`, so coefficients computed at depth `k` are
immediately available for depth `k + 1`. No quadratic recompute.

## Implementation skeleton

1. Extend `closed_branch_rel_err_bound` and `auto_eps` to consume
   `cancellation_order` (the `|v|^{−c}` amplification term).
2. Replace `max_extra = 4` with "pull until first nonzero".
3. Add the lazy order-growing loop to `auto_eps` (so `order` becomes an
   internal output, not a user parameter, for `stable_smooth`).
4. Add `stable_smooth(numerator, x, a, denominator_degree,
   cancellation_order=0)` returning an `OpFromGraph` wrapping the
   existing `taylor_remainder` machinery, with `lop_overrides` that
   builds the next-level expression via `R_n' = R_1[f' − n · R_n]`.

## Tests (status)

Implemented in `test_taylor_remainder.py`:

- `sinc(x)` iterated grads at `x = 0`. Currently asserted to depth 4
  (`sinc^(k)(0)` matches `(−1)^(k/2)/(k+1)` for k = 0..4). Beyond 4 the
  test would still pass arithmetically but compile time grows ~6× per
  level (see "Performance" below), so we cap the assertion at 4.
- `(cos(x) − 1)/x²` at `n=2`, forward + first grad.
- `(x·cos(x) − sin(x))/x²` with `cancellation_order = 2` matches
  `pt.grad(sinc(x), x)` and mpmath at 50 dps across the
  cancellation-prone neighborhood `t ∈ [1e−8, 1.0]`.
- Pitfall (1): `stable_smooth`'s grad is correct vs mpmath across
  `t ∈ [1e−12, 0.1]` where the naive quotient-rule grad of `sin(x)/x`
  underflows.
- Pitfall (2): for `expm1(x)/x`, `stable_smooth`'s grad at `x = 0`
  gives `1/2`; the naive `switch(x==0, 1, expm1(x)/x)` returns `0` via
  the constant branch (the design's pitfall).
- `float32` dtype propagation: auto-eps scales with `x.dtype`.
- `a ≠ 0` expansion: `(sin(x) − sin(a))/(x − a)` at `a = 1.7`.
- `f/g` composition: `sin(x)/tan(x) = cos(x)` via two `stable_smooth`
  calls of equal `denominator_degree=1`, divided.

## Scalar, vector, and higher-rank tensors

`stable_smooth` itself supports only scalar (`ndim=0`) and vector
(`ndim=1`) `x`.  Higher-rank tensors raise `NotImplementedError` at
the entry point: the DimShuffle whitelist in
`_check_elementwise_in_x` is broadcast-safe at rank ≤ 1 (the only
legal shuffles are expand-dims and the identity reorder), but at
rank ≥ 2 it can't distinguish axis permutations from broadcasts
without more careful analysis.

**This does not prevent elementwise application to higher-rank
tensors.**  The canonical pattern is to build `stable_smooth` over a
scalar leaf and lift with pytensor's `vectorize_graph`:

```python
import pytensor.tensor as pt
from pytensor.graph.replace import vectorize_graph
from taylor_remainder import stable_smooth

# Build once over a scalar leaf.
u = pt.dscalar("u")
sinc_u = stable_smooth(pt.sin(u), u, 0.0, denominator_degree=1)

# Apply elementwise to a tensor of any rank.
x = pt.dmatrix("x")              # or dtensor3, or dvector, ...
sinc_x = vectorize_graph(sinc_u, {u: x})
```

`vectorize_graph` is pytensor's official elementwise-lift mechanism;
gradients propagate through it correctly (verified by
`test_stable_smooth_elementwise_via_vectorize_graph`).  The narrow
`ndim ≤ 1` claim on `stable_smooth` is just an admission that the
in-place vector support's structural check is only sound at that
rank -- users who want higher-rank elementwise behavior should
delegate to `vectorize_graph`, not to our claim.

For vector `x`, the user's `numerator` must be elementwise in `x`
(each output entry depends only on the corresponding input entry):
the cache is built from a *scalar surrogate* of the numerator
produced by a rank-aware graph walk (`_scalarize_elementwise` --
needed because `clone_replace` rejects substituting a scalar where
the apply node was originally typed as vector).  The resulting cache
emits scalar `pt.constant`s that broadcast cleanly against the
vector `t = x - a` in the polynomial branch.  The pullback uses
`pt.grad(f.sum(), x)` to recover the per-entry derivative, which is
correct iff `numerator` is elementwise in `x` (diagonal Jacobian).

Non-elementwise numerators (e.g. `pt.sum(pt.sin(x)) * pt.ones_like(x)`)
are rejected up front by `_check_elementwise_in_x`, a whitelist-based
structural walk that allows only `Elemwise`, `DimShuffle`, `Alloc`,
`Shape`, `Shape_i`, `SpecifyShape` -- and only because rank ≤ 1
makes the DimShuffle case safe.  Tensor support would require a
tighter check (or a different architecture, e.g. always-scalar OFG
with `Blockwise` for vectorization).

The `cache=` cross-call sharing parameter currently requires scalar
`x` (the user can't easily produce a scalar surrogate keyed to their
vector `x`); cross-call memoization with vector inputs is future work.

## Performance

Each grad in the chain creates O(level) new OpFromGraph instances.
Construction is fast (~21 instances at depth 5).  `stable_smooth`
always builds with `OpFromGraph(inline=True)`, which on the current
pytensor (with the OFG-cloning fixes `6458acc`, `b39fced`, `a11a9b1`,
`a821179`, `7821c7e` from shortly after rel-3.0.0) avoids the
lazy-per-OFG first-eval cliff that earlier versions had.

Measured for sinc at the indicated depth (with the default
`inline=True` path):

| depth | build | `pytensor.function` | first eval | total |
|------:|------:|--------------------:|-----------:|------:|
| 3     | 0.2 s | 0.3 s               | ~0 s       | 0.6 s |
| 5     | 1.2 s | 6.4 s               | ~0 s       | 7.7 s |

The lazy-per-OFG path (formerly `inline=False`) is no longer reachable
from `stable_smooth` itself, but it can still be exercised on
`taylor_remainder`-style lower-level calls if a regression test ever
needs to.  At depth 5 it took ~115 s wall-clock total *before* the
upstream fixes; the wall-clock budget test
(`test_stable_smooth_depth5_under_wallclock_budget`, 30 s budget,
runs in ~10 s) pins the post-fix profile.

For depth ≥ ~6, `taylor_remainder_poly` is still the simpler escape
hatch when the input range stays bounded.

## Consistency checks (cosmetic, not enforced)

- `k_lead ≥ denominator_degree` is *not* required. The Taylor remainder
  construction is well-defined for any analytic f and any n: when
  `k_lead < n`, `P_{n−1}` happens to include `c_{k_lead}(x−a)^{k_lead}`
  and the subtraction works; when `k_lead ≥ n`, `P_{n−1}` is just zeros
  up to `n` and the subtraction is symbolic.
- `cancellation_order ≤ k_lead` is *not* required either.
  `exp(−1/x²)` is the counterexample — `k_lead = ∞` but `c = 0`. The two
  parameters describe entirely different facts: the series structure
  vs. the evaluation precision contract.
