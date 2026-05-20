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
    x,                        # the variable
    a=0.0,                    # expansion point
    denominator_degree,       # n in f / (x - a)^n
    cancellation_order=0,     # the user's evaluation-precision contract; see below
    # derivative_depth=0      # framework-internal, incremented by pt.grad
)
```

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
- Grad chain works correctly through depth ~5 in float64. Beyond that,
  the inner `pt.grad` across embedded `op(...)` apply nodes cascades:
  every existing op in `inner_numerator` gets a fresh child constructed,
  so compile time grows roughly 6× per level. A dedup pass that reuses
  the already-constructed child instead of building a new one would
  cap growth to linear. (Tracked as follow-up.)

## General `f / g` via composition

If both vanish at `a` to order ≥ `k` for some `k ≥ 1`:

```python
stable_smooth(f, x, a, denominator_degree=k, ...) \
    / stable_smooth(g, x, a, denominator_degree=k, ...)
```

No dedicated ratio primitive needed.

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

## Scalar only

`stable_smooth` currently raises `NotImplementedError` for non-scalar
`x`. The cache (`TaylorAtPoint`) computes coefficients via
`pt.grad(f, x)` (assumes scalar `f`) and `clone_replace(d, {x:
scalar_a})` (assumes scalar substitution). A vector-friendly version
would replace the cache with one that derives over a *scalar surrogate*
`x_s`, then broadcasts the resulting numeric coefficients across the
actual vector `x` at evaluation time. Out of scope for now.

## Performance

Each grad in the chain creates O(level) new OpFromGraph instances.
Construction itself is fast (~21 instances at depth 5; 12s at depth 4
during initial profile; 1.5s at depth 5 via `pytensor.function`).

The slow part is what happens on the first `fn(...)` call.  With the
default `inline=False`, each OpFromGraph's `_fn` (inner-function) is
compiled lazily on first call.  At depth 5 those lazy compiles
collectively cost ~30s.  Subsequent evals are ~0.1s.

Passing `inline=True` shifts the cost: the inner graphs are inlined
into the outer function during `pytensor.function` build (~50s at
depth 5), so first-eval and steady-state are essentially free.
Pytensor's graph rewriter does clone inner ops during the inline pass
(21 ops at construct time → 1077 unique ops after canonicalize at
depth 5), but the cost is paid once.

Pick `inline=True` for training loops or anything that calls the
compiled function many times.  Pick the default `inline=False` for
one-shot use or tests.  For depth ≥ ~6, `taylor_remainder_poly` is
still the simpler escape hatch when the input range stays bounded.

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
