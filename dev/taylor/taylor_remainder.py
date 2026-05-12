"""taylor_remainder defined implicitly by

    f(x)  =  P_{n-1}(x - a)  +  R(x) · (x - a)^n,

where P_{n-1} is the degree-(n-1) Taylor polynomial of f at a and R is the
n-th Taylor remainder, with R(a) = f^(n)(a)/n!. The public `taylor_remainder`
routine returns a stable graph for R evaluated across x = a.

Contract on f (the user's input).
================================

We assume the user-supplied f(x) is computed as accurately as the input
dtype allows -- i.e., f is "pristine": its evaluation has at most
~eps_machine relative error (within the dtype's representation limits).
Concretely, this means:
  - User has chosen stable libm primitives where they exist
    (e.g. `pt.expm1(x)` not `pt.exp(x) - 1`, `pt.log1p(x)` not
    `pt.log(1 + x)`).
  - User's expression doesn't introduce gratuitous cancellation
    (e.g. `pt.cos(x) - 1` is acceptable as f -- our polynomial branch
    handles the cancellation -- but `pt.exp(x) * pt.exp(-x) - 1` is not).
  - f's Taylor coefficients c_m = f^(m)(a)/m! up to m = n + order +
    (a small safety margin) are available to ~eps_machine relative
    accuracy. By
    default these are computed lazily by TaylorAtPoint via repeated
    `pt.grad`. That works for most analytic f, but the chain-rule
    expansion blows up exponentially in graph size for compositions
    like `cos(K*x**2)` where canonicalize can't consolidate the
    repeated `sin`/`cos` factors (each grad roughly doubles the graph;
    by m=14 a single derivative substitution becomes minutes). For
    such f, compute the coefficients with `mpmath.taylor` or a
    closed-form formula and pass them via the explicit-coefficients
    mode of TaylorAtPoint:
        cache = TaylorAtPoint(f, x, a, coefficients=[c_0, c_1, c_2, ...])
        # or with a lazy generator:
        cache = TaylorAtPoint(f, x, a, coefficients=my_series_iter())
        taylor_remainder(f, x, a, n, order=..., cache=cache)
    The closed branch still uses your `f` expression at runtime; only
    the polynomial-branch coefficient inventory consumed by `auto_eps`
    and the polynomial branch comes from the iterable.

Under this contract, ALL numerical error in the output of `taylor_remainder`
is introduced by the operations WE add on top of f:

  (a) The subtraction `f(x) - P_{n-1}(x - a)` (closed branch only): when
      P_{n-1} != 0 the magnitudes of the two operands are similar near a,
      causing catastrophic cancellation. PyTensor's canonicalize folds
      explicit-constant K0 - K0 patterns symbolically, which eliminates
      this for the typical case; for opaque f (e.g. wrapped in
      `OpFromGraph`) the cancellation persists at runtime.

  (b) The division by (x - a)^n (closed branch only): amplifies absolute
      errors in the numerator by 1/|x-a|^n.

  (c) Polynomial truncation at order `order` (polynomial branch only):
      contributes ~|c_{n+order}/c_n| · |x-a|^order in relative terms.

The `auto_eps` switch threshold balances (a)+(b) against (c): polynomial
branch covers the cancellation-prone region, closed branch handles
larger |x-a|. If f violates the contract (its own evaluation has more
than ~eps_machine relative error), the error from f's evaluation
propagates into the closed branch and is NOT modeled here -- the user
must either fix f or pass `eps` explicitly.

The canonicalize asymmetry.
===========================

The closed branch is built as a symbolic expression (f - P_{n-1})/t^n
and handed to PyTensor. canonicalize then folds whatever it symbolically
can. This folding is asymmetric across f's structure:

  - Polynomial f (e.g. `K0 + K1*x + K2*x^2 + K3*x^3*g(x)`): canonicalize
    folds (f - P_{n-1}) all the way down to the residual polynomial,
    eliminating runtime subtractions entirely. The closed branch is then
    computed essentially exactly, INDEPENDENT of (a)+(b) above.

  - Transcendental f (e.g. `cos(K*x)`): canonicalize folds explicit
    constants like K0 - K0 = 0 and K1*x - K1*x = 0, but cannot fold a
    transcendental call against its symbolic Taylor coefficients. The
    runtime subtraction (e.g. `cos(K*x) - 1 + K^2*x^2/2`) survives, and
    (a)+(b) apply with their full force.

This asymmetry is desirable: canonicalize gives polynomial f tighter
precision than the worst-case error model would predict. Defeating
canonicalize (by wrapping the closed branch or f itself in an opaque op)
would force the runtime cancellation in BOTH cases, degrading polynomial
f's near-exact precision to match the transcendental worst case. We
instead design auto_eps and the warnings around the worst-case error
model and let canonicalize over-deliver where it can.

The closed-cancellation gap (and the warning).
==============================================

When k_lead > n -- i.e., f's n-th derivative at a vanishes, so the
leading R coefficient sits at index k_lead > n -- the closed branch's
relative error has the form

    rel_err_closed(v)  ~  ε_m * Σ_{i ∈ [0, n)} |c_i * v^i| / (|c_{k_lead}| * v^{k_lead}),

with the 1/v^{k_lead} divisor growing rapidly as v shrinks. auto_eps's
poly-truncation-only formula does NOT account for this multi-term
cancellation, and at the resulting eps the closed branch can have
relative error well above tol_rel = 10*eps_machine. This is the case
for e.g. cos(K*x) at n=3 (c_3 = 0, k_lead = 4): with order=10, the
boundary error is ~3e-13 instead of the ~tol_rel that auto_eps's model
predicts.

`check_closed_cancellation_safety` uses `closed_branch_rel_err_bound`
to compute the expected boundary error and warns the user when it
exceeds tol_rel. The fix is to bump `order` (which widens the polynomial
window) until the gap closes, OR to pass `eps` explicitly. We do NOT
silently widen eps here because that would pull the polynomial branch
into a regime where its truncation error exceeds tol_rel for polynomial
f cases that canonicalize was successfully covering -- the asymmetry
penalizes the wrong direction.

Memoization: a TaylorAtPoint cache shares f^(m)(a) values across all calls,
so building taylor_remainder(f^(j), x, a, m) for many (j, m) pairs only
costs one chain of grads, not many.

Auto-eps: by default, eps is chosen so the polynomial truncation error is
at machine-epsilon scale (sized from x.dtype), by inspecting the first
omitted Taylor coefficient.
"""

import math
import warnings

import numpy as np

import pytensor
import pytensor.tensor as pt
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.utils import rewrite_graph


pytensor.config.on_opt_error = "ignore"


class TaylorRemainderUnderflowWarning(UserWarning):
    """The leading Taylor coefficient is small enough that the closed branch
    of taylor_remainder may underflow within the polynomial-branch window."""


class TaylorRemainderOverflowWarning(UserWarning):
    """The leading Taylor coefficient is large enough that the polynomial-
    branch leading term may overflow at the boundary |x-a| = eps."""


class TaylorRemainderClosedCancellationWarning(UserWarning):
    """The closed branch's multi-term cancellation makes its relative
    error exceed tol_rel at the auto-chosen eps. Increase `order` to
    close the gap (or pass `eps` explicitly to a preferred crossover)."""


class TaylorAtPoint:
    """Source of Taylor coefficients c_m = f^(m)(a)/m! for m = 0, 1, 2, ...

    Two modes for how coefficients are obtained:

      Auto (default, `coefficients=None`): each c_m is derived lazily from
        `f` via `pt.grad` chains (canonicalized between steps) and
        substitution of x = a. Suitable for most analytic f. Fails or
        slows badly on deeply composed transcendentals like cos(K*x**2)
        where the chain rule expands the graph exponentially -- by
        m ~= 12 a single substitution can take minutes.

      Explicit (`coefficients=<iterable of floats>`): each c_m is pulled
        from the supplied iterable on demand and converted to f^(m)(a) =
        m!*c_m for storage. The user's symbolic `f` is still used by
        `taylor_remainder`'s closed branch for runtime evaluation; only
        the coefficient inventory consumed by the polynomial branch and
        `auto_eps` comes from the iterable.

        Use this whenever you have a closed-form series (cos, exp, etc.)
        or can compute coefficients with `mpmath.taylor` faster than
        pt.grad. The iterable can be a list (eager) or a generator
        (lazy, infinite-friendly); pulled with `next()` as the cache
        needs new indices, and an `IndexError` is raised if it runs out.

    Auto-mode coefficients are computed and memoized lazily, so reuse
    across multiple `taylor_remainder` calls with the same (f, x, a)
    avoids redundant grad+canonicalize work.
    """

    def __init__(self, f, x, a, *, coefficients=None):
        self.f = f
        self.x = x
        self.a = a
        self._a_const = pt.constant(a, dtype=x.dtype)
        self._derivs = [f]
        self._values = []  # f^(m)(a) as symbolic constants
        self._numeric = {}  # m -> float(f^(m)(a))
        self._coeff_source = iter(coefficients) if coefficients is not None else None
        # Running float factorial used by the explicit-coefficients path
        # to convert c_k -> f^(k)(a) = k!*c_k. Stays in float arithmetic
        # so order > 18 (where math.factorial exceeds 2^53 and loses
        # precision under int->float conversion) stays correct.
        self._running_factorial = 1.0

    def deriv(self, m):
        """The m-th derivative of f as a symbolic pytensor expression.

        In auto mode, this is the exact symbolic derivative of the user's
        f via repeated `pt.grad` (chain rule), canonicalized between steps.

        In explicit-coefficients mode, the user has implicitly defined f
        by its Taylor series coefficients [c_0, c_1, c_2, ...], so we
        build the polynomial truncation
            f_poly(x)  =  Σ_{k=0..N}  c_k · (x - a)^k
        (where N is the highest index pulled from the iterable so far)
        and take the m-th derivative of that.  The result is a symbolic
        pytensor expression -- exact at x=a (matches m!·c_m), and
        polynomial-accurate near x=a.  If the iterable is exhausted
        before c_m is reached, raises IndexError (no silent zero-pad).
        """
        if self._coeff_source is not None:
            # Pull coefficients up to c_m (may raise IndexError).
            self.value_at_a(m)
            # The m-th derivative of f_poly(x) = Σ_{k=0..N} c_k · (x-a)^k
            # has derivative coefficients
            #     b_j = c_{j+m} · (j+m)! / j!  =  numeric[j+m] / j!
            # for j ∈ [0, N-m]. Construct directly by Horner (same
            # precision guarantees as the polynomial branch in
            # taylor_remainder), no pt.grad chain needed.
            N = len(self._values) - 1
            D = N - m  # degree of the m-th derivative polynomial
            t = self.x - self._a_const
            # Use a running float product for j!, not math.factorial(j),
            # to avoid (a) int -> float precision loss for j > 18 (where
            # math.factorial exceeds 2^53) and (b) recomputing the
            # factorial from scratch each Horner step.
            fac = 1.0
            for k in range(1, D + 1):
                fac *= k  # fac = D!
            # Highest term: b_D = numeric[N] / D!
            poly = pt.constant(self._numeric[N] / fac, dtype=self.x.dtype)
            # Horner down from j = D-1 to j = 0, updating fac incrementally
            # from (j+1)! to j!.
            for j in range(D - 1, -1, -1):
                fac /= j + 1
                b_j = self._numeric[j + m] / fac
                poly = poly * t + b_j
            return poly

        while len(self._derivs) <= m:
            try:
                d = pt.grad(self._derivs[-1], self.x)
            except pytensor.gradient.DisconnectedInputError:
                # f^(k) is a constant (e.g. f is a polynomial); higher
                # derivatives are all zero.
                d = pt.constant(0.0, dtype=self.x.dtype)
            d = rewrite_graph(d, include=("canonicalize",))
            self._derivs.append(d)
        return self._derivs[m]

    def value_at_a(self, m):
        while len(self._values) <= m:
            k = len(self._values)
            if self._coeff_source is not None:
                try:
                    c_next = next(self._coeff_source)
                except StopIteration as exc:
                    raise IndexError(
                        f"TaylorAtPoint: explicit coefficient iterable exhausted "
                        f"at m={k}; need at least m={m + 1} coefficients"
                    ) from exc
                if k > 0:
                    self._running_factorial *= k
                v_m = self._running_factorial * float(c_next)
                self._values.append(pt.constant(v_m, dtype=self.x.dtype))
                self._numeric[k] = v_m
            else:
                d = self.deriv(k)
                v = clone_replace(d, {self.x: self._a_const})
                v = rewrite_graph(v, include=("canonicalize",))
                self._values.append(v)
        return self._values[m]

    def numeric_value_at_a(self, m):
        """f^(m)(a) as a Python float.  Cached -- only compiled once per m."""
        if m not in self._numeric:
            self._numeric[m] = float(self.value_at_a(m).eval())
        return self._numeric[m]

    def coeff(self, m):
        """f^(m)(a) / m!  (symbolic constant)."""
        v = self.value_at_a(m)
        fac = math.factorial(m)
        if fac == 1:
            # Avoid spurious True_div(v, 1.0) -- canonicalize folds it under
            # FAST_RUN, but FAST_COMPILE leaves it in place, breaking the
            # structural match between the K0 inside f and the K0 inside P.
            return v
        return v / float(fac)

    def numeric_coeff(self, m):
        """f^(m)(a) / m!  as a Python float."""
        return self.numeric_value_at_a(m) / math.factorial(m)

    def coeffs_of_deriv(self, j, K):
        """Taylor coefficients [c_0, ..., c_{K-1}] of f^(j) at a.

        c_l = (f^(j))^(l)(a) / l! = f^(j+l)(a) / l!
        """
        return [self.coeff(j + l) for l in range(K)]


# Cap on the "pull until nonzero" scan in `_first_nonvanishing`.  Plenty
# of headroom for any realistic sparsity pattern (parity-N etc.), bounded
# to prevent infinite loops on pathological cases like exp(-1/x^2) whose
# Taylor coefficients at 0 are all zero despite the function being smooth.
_FIRST_NONVANISHING_SAFETY = 64


def _first_nonvanishing(cache, start, count=_FIRST_NONVANISHING_SAFETY):
    """Return (|c_k|, k) for the first k in [start, start+count) with c_k != 0.

    Compares against 0 exactly: pytensor's canonicalize folds mathematically
    zero expressions to literal zero, so any nonzero result -- however small --
    represents a genuine nonzero coefficient (e.g. 1e-50 from a scaled f).

    `count` defaults to `_FIRST_NONVANISHING_SAFETY` -- pulls coefficients
    one at a time until a nonzero is found, with a safety bound that
    handles realistic sparsity. Callers pass a smaller `count` when they
    have a specific bounded window in mind (e.g. "is c_lead nonzero
    anywhere in the polynomial branch?", where the window is `order`).

    Use this for v_lead-style scans where exhaustion of an explicit-
    coefficient cache should raise (the user under-provisioned). For
    v_trunc-style scans past the polynomial end, use
    `_first_nonvanishing_past_polynomial`, which treats exhaustion as
    "all remaining coefficients are zero".
    """
    for k in range(start, start + count):
        v = abs(cache.numeric_coeff(k))
        if v != 0.0:
            return v, k
    return 0.0, start + count


def _first_nonvanishing_past_polynomial(cache, start, count=_FIRST_NONVANISHING_SAFETY):
    """Variant of `_first_nonvanishing` for v_trunc-style scans.

    Same semantics, except that exhaustion of an explicit-coefficient
    cache returns (0.0, start) instead of raising IndexError. The
    rationale: when the cache is finite (e.g. the user supplied
    [c_0, ..., c_N] for a polynomial f), there genuinely are no
    nonzero coefficients past the list, and the truncation budget is
    "all remaining zero" -- valid input, not an error.
    """
    try:
        return _first_nonvanishing(cache, start, count)
    except IndexError:
        return 0.0, start


def _tol_rel(order, dtype):
    """Target relative error for auto_eps's branch-balancing.

    Set to 2x `poly_branch_rel_err_bound(order)` -- twice the
    polynomial-branch rounding floor -- so that the analytical eps
    formula leaves equal budget for poly truncation and poly-evaluation
    rounding. With this choice, at the chosen eps:
        rel_err_total  ≈  trunc(eps) + poly_rounding
                        ≈  tol_rel/2 + tol_rel/2  =  tol_rel
    and `check_closed_cancellation_safety` warns iff the closed branch
    *adds* more error than this whole budget covers.

    The 2x multiplier is the only design constant here; using 1x would
    tie poly truncation = poly rounding exactly with no slack, and
    using a larger constant inflates the "acceptable" bound without
    physical justification.
    """
    return 2.0 * poly_branch_rel_err_bound(order, dtype=dtype)


def auto_eps(cache, n, order, *, dtype=None):
    """Threshold for switching from polynomial branch to closed-form branch.

    The polynomial truncation relative error at |t|=eps is approximately
        |c_{n+order} / c_n| · eps^order
    (or the first nonzero term beyond order if c_{n+order}=0; or, if the
    leading c_n vanishes too, the first nonzero c_k for k>=n). The formula
    is scale-invariant in f -- multiplying f by a constant doesn't change
    eps, since the ratio of coefficients is unchanged.

    We pick eps so this relative error reaches `tol_rel`, the target set
    by `_tol_rel` -- 2x the polynomial-branch rounding floor, derived from
    Wilkinson's bound. No empirical safety factor: under the principled
    tol_rel, the analytical formula already includes the necessary margin.

    Two regimes:

    1. v_trunc != 0 (formula path): polynomial truncation gives the
       binding constraint. The closed branch's potential cancellation
       in (f - P_{n-1}) is left to pytensor's canonicalize, which folds
       explicit-constant K0 - K0 patterns symbolically. (Adversarial
       opaque f, e.g. inside an OpFromGraph, can defeat canonicalize;
       in those cases pass `eps` explicitly.)

    2. v_trunc = 0 (no truncation budget): fall back to the
       cancellation-aware lower bound from the closed-branch error model
       err_closed ≈ eps_machine · |c_0|/(|c_{k_lead}| · |x-a|^{k_lead}).
       Returns eps = (eps_machine · |c_0|/(tol_rel · |c_{k_lead}|))^(1/k_lead),
       using k_lead (not n) so that sparse-leading-coefficient cases
       (c_n = 0, e.g. f = K_0 + K_2·x^2 at n=1 has k_lead = 2) get the
       correct exponent. Returns 0 when P_{n-1} is symbolically zero
       (no subtraction introduced).

    The formula is NOT capped at any constant -- doing so would violate
    scale invariance (the natural cap is the function's convergence radius,
    which lives in the units of x).

    `dtype` defaults to the dtype of the cache's input variable.

    Empirical validation: dev/taylor/taylor_eps_experiment.py.
    """
    if dtype is None:
        dtype = np.dtype(cache.x.dtype)
    else:
        dtype = np.dtype(dtype)
    eps_machine = float(np.finfo(dtype).eps)
    tol_rel = _tol_rel(order, dtype)

    # Leading coefficient of R: first nonzero c_k for k >= n (typically c_n).
    v_lead, k_lead = _first_nonvanishing(cache, n, order)
    if v_lead == 0.0:
        # R itself effectively vanishes in our coefficient window. Closed
        # form is also zero (numerator and denominator both vanish). The
        # switch in taylor_remainder substitutes the polynomial limit
        # (which is zero) at x=a; closed is used elsewhere. eps=0 means
        # this fall-through.
        return 0.0

    # The binding constraint depends on whether the polynomial-truncation
    # budget is finite.
    #
    # v_trunc != 0  (typical case):
    #   Polynomial truncation gives  eps <= eps_upper  for poly accuracy.
    #   Closed-branch cancellation in (f - P_{n-1}) is handled at runtime
    #   by pytensor's canonicalize: when f's structure makes the K0 - K0
    #   subtraction explicit-constant cancellation, canonicalize folds it
    #   symbolically. We trust this for the formula path. (Adversarial
    #   opaque f -- e.g. wrapped in OpFromGraph -- can defeat canonicalize;
    #   for those, the user can pass `eps` explicitly.)
    #
    # v_trunc = 0  (corner case, e.g. polynomial f or extreme subnormal):
    #   No polynomial-truncation bound. Fall back to a cancellation-aware
    #   lower bound on eps from
    #       err_closed(eps) ≈ eps_machine · |c_0|/(|c_{k_lead}| · eps^{k_lead})
    #   so that even if canonicalize doesn't fold, closed accuracy is
    #   maintained for |x-a| >= eps. When c_0 = ... = c_{n-1} = 0
    #   (P_{n-1} symbolically zero), no subtraction is introduced and
    #   eps_lower = 0 (closed is fine for any x != a).

    v_trunc, k_trunc = _first_nonvanishing_past_polynomial(cache, n + order)
    if v_trunc == 0.0:
        if n == 0:
            return 0.0
        v_const, _ = _first_nonvanishing(cache, 0, n)
        if v_const == 0.0:
            return 0.0
        return (eps_machine * v_const / (tol_rel * v_lead)) ** (1.0 / k_lead)

    # NB: compute the ratio v_lead/v_trunc *before* multiplying by tol_rel
    # to avoid floating-point underflow when v_lead is subnormal -- the
    # ratio is K-invariant, but `tol_rel * v_lead` underflows for K
    # near smallest_subnormal.
    return (tol_rel * (v_lead / v_trunc)) ** (1.0 / (k_trunc - k_lead))


def check_underflow_safety(cache, n, eps, *, dtype=None, safety=10.0):
    """Issue a warning if the closed branch may underflow within |x| < eps.

    The closed branch of taylor_remainder evaluates  (f(x) - P_{n-1}(x-a))/(x-a)^n
    at  |x-a| >= eps. Near the boundary, this magnitude behaves like
    |c_n| · eps^n. If that quantity is below `safety · smallest_subnormal`, the
    user's f(x) computation is at risk of underflowing to zero, returning 0
    where the true value is c_n.

    No warning is issued when c_n = 0 (f vanishes to higher order than n;
    the user's expression doesn't have the leading c_n·x^n term to underflow).

    See `check_overflow_safety` for the symmetric large-|c_n| condition.
    """
    if dtype is None:
        dtype = np.dtype(cache.x.dtype)
    else:
        dtype = np.dtype(dtype)
    smallest_subnormal = float(np.finfo(dtype).smallest_subnormal)

    try:
        c_n = abs(cache.numeric_coeff(n))
    except Exception:
        return
    if c_n == 0.0:
        return

    closed_magnitude = c_n * eps**n
    threshold = safety * smallest_subnormal
    if closed_magnitude < threshold:
        warnings.warn(
            f"taylor_remainder: leading coefficient |c_{n}| = {c_n:.3g} is small "
            f"enough that the closed branch may underflow within the polynomial-"
            f"branch window |x-a| < {eps:.3g}.  At |x-a|=eps, the closed-branch "
            f"numerator scale is |c_{n}|·eps^{n} = {closed_magnitude:.2e}, below "
            f"the safety threshold {safety}·smallest_subnormal = {threshold:.2e}.  "
            f"Consider raising `order` to widen `eps`, passing a larger `eps` "
            f"explicitly, or using `taylor_remainder_poly` if x stays bounded.",
            TaylorRemainderUnderflowWarning,
            stacklevel=3,
        )


def check_overflow_safety(cache, n, eps, *, dtype=None, safety=10.0):
    """Issue a warning if the polynomial-branch leading term may overflow.

    Symmetric to `check_underflow_safety`. The polynomial branch evaluates
    a sum dominated by  c_n + O(eps)  near the boundary |x-a| = eps; more
    generally the closed-branch numerator scales as  |c_n| · eps^n.
    If that quantity exceeds `largest_finite / safety`, evaluation of the
    closed-branch numerator (or the polynomial sum) may overflow to inf.

    Whether this can fire depends on whether eps > 1 is achievable. Auto_eps
    tends to give eps < 1 for float64 with moderate `order`, but at high order
    or low precision (float32, float16) eps can exceed 1 -- e.g. we measure
    auto_eps ≈ 2 for sin/expm1 in float32 at order=14. Combined with
    |c_n| close to largest_finite, the polynomial-branch leading term overflows.

    No warning is issued when c_n = 0.
    """
    if dtype is None:
        dtype = np.dtype(cache.x.dtype)
    else:
        dtype = np.dtype(dtype)
    largest_finite = float(np.finfo(dtype).max)

    try:
        c_n = abs(cache.numeric_coeff(n))
    except Exception:
        return
    if c_n == 0.0:
        return

    closed_magnitude = c_n * eps**n
    threshold = largest_finite / safety
    if not math.isfinite(closed_magnitude) or closed_magnitude > threshold:
        warnings.warn(
            f"taylor_remainder: leading coefficient |c_{n}| = {c_n:.3g} is large "
            f"enough that the polynomial-branch leading term may overflow at the "
            f"boundary |x-a| = {eps:.3g}.  At |x-a|=eps, the closed-branch "
            f"numerator scale is |c_{n}|·eps^{n} = {closed_magnitude:.2e}, above "
            f"the safety threshold largest_finite/{safety} = {threshold:.2e}.  "
            f"Consider rescaling f, passing a smaller `eps` explicitly, or "
            f"reducing `order`.",
            TaylorRemainderOverflowWarning,
            stacklevel=3,
        )


def _subtracted_terms_combined_magnitude(cache, n, v):
    """Σ_{i ∈ [0, n)} |c_i · v^i|: the combined magnitude of operands
    entering the subtraction (f - P_{n-1}) at |x-a| = v.

    By Wilkinson's bound for floating-point summation, computing
    `f(v) - Σ c_i·v^i` gives an absolute error bounded by
        ε_m · (|f(v)| + Σ|c_i·v^i|)  +  higher-order accumulation,
    and within the radius of convergence |f(v)| ≤ Σ_{i=0}^∞ |c_i·v^i|,
    so this combined sum (extended formally to all of f's series, but
    truncated to [0,n) for the cache values we have) is the leading
    term in the bound. We omit the higher-order n·ε_m accumulation
    factor; empirically that overcounts the rounding errors actually
    realized in fp arithmetic.

    Returns 0 when P_{n-1} ≡ 0 symbolically (c_0 = ... = c_{n-1} = 0),
    in which case no subtraction is introduced and there is no
    cancellation contribution.
    """
    total = 0.0
    for i in range(n):
        try:
            mag = abs(cache.numeric_coeff(i)) * (abs(v) ** i if i > 0 else 1.0)
        except Exception:
            mag = 0.0
        total += mag
    return total


def closed_branch_rel_err_bound(
    cache, n, v, order, *, dtype=None, cancellation_order=0
):
    """Upper bound on the closed branch's relative error at |x-a| = v.

    Derivation.  Under the contract that f's computed value has
    relative error  ≤ ε_m · |v|^{-c}  (where c = `cancellation_order`,
    default 0 = pristine evaluation), two contributions enter the
    closed branch's relative error:

      - The P_{n-1} subtraction (Wilkinson):  abs_err ≤ ε_m · combined
        where  combined = Σ_{i ∈ [0,n)} |c_i · v^i|.  Translated to
        the result's relative scale by dividing by  |c_{k_lead}|·v^{k_lead}:
        rel contribution =  ε_m · combined / (|c_{k_lead}| · v^{k_lead}).

      - f's own evaluation plus the divide-by-v^n rounding:  baseline
        ε_m for pristine f, amplified to  ε_m · |v|^{-c}  for c > 0.

    Total:

        rel_err_closed(v; c)
            ≤  ε_m · combined / (|c_{k_lead}| · v^{k_lead})       # P-subtraction
             + ε_m · max(1, |v|^{-c})                              # f's eval + division

    For c = 0 (pristine), the second term collapses to ε_m and we
    recover the existing pristine-f formula  ε_m · (cancellation + 1).
    For c > 0 and |v| < 1, the |v|^{-c} factor amplifies the floor,
    stretching `eps_lower` outward exactly as the user's evaluation
    contract specifies.

    Two regimes worth naming (independent of c):
      - k_lead == n (typical): the formula reduces to the single-term
        cancellation model.
      - k_lead > n (c_n vanishes -- e.g. cos(Kx) at n=3 has c_3=0,
        k_lead=4): the 1/v^{k_lead} divisor amplifies precision loss
        as v shrinks, giving a 1/v^{k_lead-n} factor on top of the
        single-term scale. auto_eps does NOT account for this, so
        check_closed_cancellation_safety warns the user.

    Returns ∞ when v=0 (closed branch undefined) or when R itself
    vanishes in the cache's coefficient window.
    """
    if v == 0:
        return float("inf")
    v_lead, k_lead = _first_nonvanishing(cache, n, order)
    if v_lead == 0.0:
        return float("inf")
    if dtype is None:
        dtype = np.dtype(cache.x.dtype)
    eps_machine = float(np.finfo(np.dtype(dtype)).eps)
    combined = _subtracted_terms_combined_magnitude(cache, n, abs(v))
    cancellation = combined / (v_lead * abs(v) ** k_lead) if combined > 0 else 0.0
    floor = max(1.0, abs(v) ** (-cancellation_order)) if cancellation_order > 0 else 1.0
    return eps_machine * (cancellation + floor)


def poly_branch_rel_err_bound(order, *, dtype=None, cache=None, n=None, v=None):
    """Upper bound on the polynomial branch's relative error.

    Has two sources, both bounded:

      rounding:    (2·order + 1)·ε_m   (Wilkinson on Horner-method, `order`
                   multiplications + `order` additions plus the final result;
                   `taylor_remainder` builds the polynomial by Horner since
                   the same commit that added this documentation)
      truncation:  |c_{k_trunc}/c_{k_lead}| · v^{k_trunc - k_lead}
                   (relative magnitude of the first omitted term, from the
                   cache's coefficient inventory)

    If `cache`, `n`, `v` are all provided, returns rounding + truncation.
    Otherwise returns rounding alone (for use when the truncation budget
    is irrelevant, e.g. inside the polynomial branch's interior).
    """
    if dtype is None:
        dtype = np.float64 if cache is None else cache.x.dtype
    eps_machine = float(np.finfo(np.dtype(dtype)).eps)
    rounding = (2 * order + 1) * eps_machine
    if cache is None or n is None or v is None or v == 0:
        return rounding
    v_lead, k_lead = _first_nonvanishing(cache, n, order)
    if v_lead == 0.0:
        return rounding
    v_trunc, k_trunc = _first_nonvanishing_past_polynomial(cache, n + order)
    if v_trunc == 0.0:
        return rounding
    truncation = (v_trunc / v_lead) * abs(v) ** (k_trunc - k_lead)
    return rounding + truncation


def check_closed_cancellation_safety(cache, n, eps, order, *, dtype=None):
    """Issue a warning if multi-term cancellation makes the closed branch's
    boundary error exceed tol_rel.

    The bound is computed by `closed_branch_rel_err_bound` (which captures
    both single-term and multi-term cancellation). The warning fires when
    k_lead > n (i.e., c_n = 0 -- f's n-th derivative vanishes at a) AND
    the predicted boundary error exceeds tol_rel = 10·ε_m. For k_lead = n
    (typical case) auto_eps's polynomial-truncation formula already
    balances closed-branch single-term cancellation against poly
    truncation; no warning is needed.
    """
    if dtype is None:
        dtype = np.dtype(cache.x.dtype)
    else:
        dtype = np.dtype(dtype)
    eps_machine = float(np.finfo(dtype).eps)
    tol_rel = _tol_rel(order, dtype)

    v_lead, k_lead = _first_nonvanishing(cache, n, order)
    if v_lead == 0.0 or k_lead == n:
        return  # no multi-term cancellation
    combined = _subtracted_terms_combined_magnitude(cache, n, eps)
    if combined == 0.0:
        return  # P_{n-1} ≡ 0 symbolically, no subtraction
    if eps <= 0.0:
        return

    rel_err_at_boundary = closed_branch_rel_err_bound(cache, n, eps, order, dtype=dtype)
    if rel_err_at_boundary <= tol_rel:
        return

    eps_lower = (eps_machine * combined / (tol_rel * v_lead)) ** (1.0 / k_lead)
    warnings.warn(
        f"taylor_remainder: closed branch's multi-term cancellation predicts "
        f"relative error ~{rel_err_at_boundary:.2e} at |x-a|=eps, exceeding "
        f"tol_rel={tol_rel:.2e}.  This arises because c_{n}=0 (k_lead={k_lead}>n={n}), "
        f"so the numerator (f - P_{{n-1}}) is built by subtracting terms of combined "
        f"magnitude ~{combined:.3g} to obtain a result of magnitude ~|c_{{{k_lead}}}|·eps^{k_lead}"
        f"={v_lead * eps**k_lead:.3g}.  The polynomial branch covers x < eps={eps:.3g} "
        f"with rel_err <= tol_rel; the closed branch only reaches tol_rel for "
        f"|x| >= {eps_lower:.3g}.  Increase `order` to widen the polynomial window "
        f"and close the gap, or pass `eps` explicitly.",
        TaylorRemainderClosedCancellationWarning,
        stacklevel=3,
    )


def closed_branch_needed(cache, n, order, t_max, *, dtype=None):
    """Return True if the polynomial branch alone is insufficient over |t| <= t_max.

    When False, you can drop the closed branch entirely -- polynomial achieves
    `tol_rel`-level relative accuracy throughout the input range, saving a
    transcendental call per element.
    """
    if dtype is None:
        dtype = np.dtype(cache.x.dtype)
    else:
        dtype = np.dtype(dtype)
    tol_rel = _tol_rel(order, dtype)

    v_lead, k_lead = _first_nonvanishing(cache, n, order)
    if v_lead == 0.0:
        return False
    v_trunc, k_trunc = _first_nonvanishing_past_polynomial(cache, n + order)
    if v_trunc == 0.0:
        return False
    # relative truncation at t_max: |c_trunc / c_lead| · t_max^(k_trunc - k_lead)
    return (v_trunc / v_lead) * t_max ** (k_trunc - k_lead) > tol_rel


def taylor_remainder(f, x, a, n, *, order=10, eps=None, dtype=None, cache=None):
    """Numerically stable evaluation of the n-th Taylor remainder of f at a.

    The n-th Taylor remainder R is defined by

        f(x)  =  P_{n-1}(x - a)  +  R(x) · (x - a)^n,

    where P_{n-1} is the degree-(n-1) Taylor polynomial of f at a:

        P_{n-1}(t)  =  Σ_{k=0..n-1}  f^(k)(a) / k!  ·  t^k.

    Equivalently, R(x) = (f(x) - P_{n-1}(x-a)) / (x-a)^n -- but that closed
    form has 0/0 at x=a, with R(a) = f^(n)(a) / n!. This routine returns
    a graph that evaluates R stably across x=a.

    Internally evaluated as

        switch(|x - a| < eps,
               polynomial branch (degree order-1 truncation of R's series),
               closed form (f(x) - P_{n-1}(x-a)) / (x-a)^n)

    Parameters
    ----------
    f : Variable
        Symbolic expression in `x`.
    x : Variable
        Input variable. Its dtype is fixed at variable creation and is
        used to size the auto-chosen eps.
    a : float
        Expansion point.
    n : int
        Order of the Taylor remainder.
    order : int, default 10
        Polynomial-branch length. Must exceed the number of derivatives you
        plan to take through the resulting graph (each grad pass shortens
        the polynomial branch by one).
    eps : float, optional
        Switch threshold. If None, chosen by `auto_eps` to drive polynomial
        truncation error to ~10·eps_machine for the relevant dtype. The
        chosen value is a Python float baked into the graph.
    dtype : dtype, optional
        Override for the dtype used to size auto-chosen eps. Defaults to
        `x.dtype`. Ignored when `eps` is given explicitly.
    cache : TaylorAtPoint, optional
        Shared cache of f^(m)(a) values. Pass one to memoize derivative
        evaluations across multiple taylor_remainder calls.
    """
    if cache is None:
        cache = TaylorAtPoint(f, x, a)
    coeffs = cache.coeffs_of_deriv(0, n + order)
    t = x - a

    # Horner-method evaluation:  poly = ((c_{n+order-1}·t + c_{n+order-2})·t + ...) · t + c_n
    # Wilkinson's bound for this is (2·order + 1)·ε_m, captured in
    # poly_branch_rel_err_bound. The previous direct-sum form (each c_k·t^k
    # computed independently and added) had a strictly looser O(order²)
    # bound and could exceed (2·order+1)·ε_m on adversarial inputs.
    poly = coeffs[n + order - 1]
    for k in range(order - 2, -1, -1):
        poly = poly * t + coeffs[n + k]

    if n == 0:
        closed = f
    else:
        # Same Horner construction for the polynomial we subtract from f.
        if n == 1:
            P = coeffs[0]
        else:
            P = coeffs[n - 1]
            for k in range(n - 2, -1, -1):
                P = P * t + coeffs[k]
        closed = (f - P) / t**n

    if eps is None:
        eps = auto_eps(cache, n, order, dtype=dtype)

    if eps == 0:
        # Degenerate case: no truncation budget could be evaluated.
        # Use polynomial branch with switch -- closed branch is the user's f
        # which captures any beyond-window terms; we only fall back to the
        # polynomial limit value at exactly x=a (where closed has 0/0).
        # For c_0 != 0 the subtraction in (f - P_{n-1}) suffers cancellation
        # at small |x-a|; the cancellation-aware eps is computed via auto_eps
        # path returning a positive eps, not 0. So eps=0 here means R itself
        # vanishes in our window (v_lead = 0) or the user explicitly set 0.
        return pt.switch(pt.eq(t, 0), poly, closed)

    check_underflow_safety(cache, n, eps, dtype=dtype)
    check_overflow_safety(cache, n, eps, dtype=dtype)
    check_closed_cancellation_safety(cache, n, eps, order, dtype=dtype)

    return pt.switch(pt.abs(t) < eps, poly, closed)


def taylor_remainder_poly(f, x, a, n, *, order=10, cache=None):
    """Polynomial-only approximation of the n-th Taylor remainder of f at a.

    Truncates the power series of R from
        f(x)  =  P_{n-1}(x - a)  +  R(x) · (x - a)^n
    to its first `order` terms, returning

        Σ_{k=0..order-1}  f^(k+n)(a) / (k+n)!  ·  (x - a)^k.

    Equals `taylor_remainder` up to a truncation error O((x-a)^order). Use
    this when (x-a) stays small enough that the closed-form branch is
    unnecessary -- it's faster, has trivial gradients, and removes the switch.
    """
    if cache is None:
        cache = TaylorAtPoint(f, x, a)
    coeffs = cache.coeffs_of_deriv(0, n + order)
    t = x - a
    return sum(coeffs[n + k] * t**k for k in range(order))


def main():
    x = pt.dscalar("x")

    # Auto-chosen eps for three test functions and a closed-branch necessity
    # check at |t| <= 0.5.
    funcs = [
        ("log1p(x)/x     ", pt.log1p(x), 1),
        ("exp(-x^2/2)    ", pt.exp(-(x**2) / 2.0), 0),
        ("(cos(x)-1)/x^2 ", pt.cos(x) - 1, 2),
    ]
    for name, f_expr, n in funcs:
        cache_ = TaylorAtPoint(f_expr, x, 0.0)
        for order in (8, 10, 12, 14):
            eps_ = auto_eps(cache_, n, order)
            need = closed_branch_needed(cache_, n, order, t_max=0.5)
            print(
                f"  {name}  n={n}  order={order:>2}  "
                f"auto_eps={eps_:>8.3g}  closed_branch_needed(|t|<=0.5)={need}"
            )
        print()

    # Iterated grad of log1p / x with shared cache.
    print("\n=== iterated grad of taylor_remainder(log1p, x, 0, 1) ===")
    cache = TaylorAtPoint(pt.log1p(x), x, 0.0)
    cur = taylor_remainder(pt.log1p(x), x, 0.0, 1, order=10, cache=cache)
    cur = rewrite_graph(cur, include=("canonicalize",))

    def ref(k):
        return (-1) ** k * math.factorial(k) / (k + 1)

    for k in range(8):
        fn = pytensor.function([x], cur)
        v = float(fn(0.0))
        r = ref(k)
        ok = "ok" if abs(v - r) <= 1e-9 * max(1.0, abs(r)) else "FAIL"
        print(f"  k={k}  {ok}  val={v:>16.10g}  ref={r:>16.10g}")
        cur = pt.grad(cur, x)
        cur = rewrite_graph(cur, include=("canonicalize",))


if __name__ == "__main__":
    main()
