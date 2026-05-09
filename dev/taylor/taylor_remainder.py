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
  - f's symbolic derivatives f^(m)(a) are computable to arbitrary order
    m (we need up to m = n + order + max_extra) and each evaluates to
    ~eps_machine relative accuracy at x=a. PyTensor handles the symbolic
    differentiation via `pt.grad` chains, with canonicalize simplifying
    each derivative; for most analytic f this works, but pathological
    cases (e.g. `cos(K*x^2)` whose grad chain blows up pytensor's
    recursion limit, or expressions whose derivatives canonicalize fails
    to simplify) violate the contract. In those cases the user must
    pre-compute f^(m)(a) and pre-populate the TaylorAtPoint cache rather
    than relying on the lazy grad chain.

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
    """Memoizes f, f', f'', ... and f^(m)(a) for m = 0, 1, 2, ...

    Each value is built lazily on demand. Reuse across multiple taylor_remainder
    calls with the same (f, x, a) avoids redundant grad+canonicalize work.
    """

    def __init__(self, f, x, a):
        self.f = f
        self.x = x
        self.a = a
        self._a_const = pt.constant(a, dtype=x.dtype)
        self._derivs = [f]
        self._values = []  # f^(m)(a) as symbolic constants
        self._numeric = {}  # m -> float(f^(m)(a))

    def deriv(self, m):
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
            d = self.deriv(len(self._values))
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


def _first_nonvanishing(cache, start, count):
    """Return (|c_k|, k) for the first k in [start, start+count) with c_k != 0.

    Compares against 0 exactly: pytensor's canonicalize folds mathematically
    zero expressions to literal zero, so any nonzero result -- however small --
    represents a genuine nonzero coefficient (e.g. 1e-50 from a scaled f).
    """
    for k in range(start, start + count):
        try:
            v = abs(cache.numeric_coeff(k))
        except Exception:
            v = 0.0
        if v != 0.0:
            return v, k
    return 0.0, start + count


def auto_eps(cache, n, order, *, dtype=None, safety=0.75, max_extra=4):
    # TODO: `max_extra=4` is a heuristic for finding the next nonzero
    # coefficient when c_{n+order} = 0. It works for series with regularly-
    # placed zeros (e.g., even/odd parity) but could miss pathologically
    # sparse series with longer runs of zero coefficients. Revisit if such
    # cases come up.
    """Threshold for switching from polynomial branch to closed-form branch.

    The polynomial truncation relative error at |t|=eps is approximately
        |c_{n+order} / c_n| · eps^order
    (or the first nonzero term beyond order if c_{n+order}=0; or, if the
    leading c_n vanishes too, the first nonzero c_k for k>=n). The formula
    is scale-invariant in f -- multiplying f by a constant doesn't change
    eps, since the ratio of coefficients is unchanged.

    We pick eps so this relative error reaches ~10·eps_machine for the
    given dtype, then apply a `safety` factor (default 0.75) to account
    for the empirical ~25% overshoot of the analytic formula.

    Two regimes:

    1. v_trunc != 0 (formula path): polynomial truncation gives the
       binding constraint. The closed branch's potential cancellation
       in (f - P_{n-1}) is left to pytensor's canonicalize, which folds
       explicit-constant K0 - K0 patterns symbolically. (Adversarial
       opaque f, e.g. inside an OpFromGraph, can defeat canonicalize;
       in those cases pass `eps` explicitly.)

    2. v_trunc = 0 (no truncation budget): fall back to the
       cancellation-aware lower bound from the closed-branch error model
       err_closed ≈ eps_machine · |c_0|/(|c_n| · |x-a|^n). Returns
       eps = (eps_machine · |c_0|/(tol_rel · |c_n|))^(1/n), or 0 when
       P_{n-1} is symbolically zero (no subtraction introduced).

    The formula is NOT capped at any constant -- doing so would violate
    scale invariance (the natural cap is the function's convergence radius,
    which lives in the units of x).

    `dtype` defaults to the dtype of the cache's input variable.

    Empirical validation: dev/taylor/taylor_eps_experiment.py,
                          dev/taylor/safety_calibration.py.
    """
    if dtype is None:
        dtype = np.dtype(cache.x.dtype)
    else:
        dtype = np.dtype(dtype)
    eps_machine = float(np.finfo(dtype).eps)
    tol_rel = 10.0 * eps_machine

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
    #       err_closed(eps) ≈ eps_machine · |c_0|/(|c_n| · eps^n)
    #   so that even if canonicalize doesn't fold, closed accuracy is
    #   maintained for |x-a| >= eps. When c_0 = ... = c_{n-1} = 0
    #   (P_{n-1} symbolically zero), no subtraction is introduced and
    #   eps_lower = 0 (closed is fine for any x != a).

    v_trunc, k_trunc = _first_nonvanishing(cache, n + order, max_extra + 1)
    if v_trunc == 0.0:
        if n == 0:
            return 0.0
        v_const, _ = _first_nonvanishing(cache, 0, n)
        if v_const == 0.0:
            return 0.0
        return (eps_machine * v_const / (tol_rel * v_lead)) ** (1.0 / n)

    # NB: compute the ratio v_lead/v_trunc *before* multiplying by tol_rel
    # to avoid floating-point underflow when v_lead is subnormal -- the
    # ratio is K-invariant, but `tol_rel * v_lead` underflows for K
    # near smallest_subnormal.
    return safety * (tol_rel * (v_lead / v_trunc)) ** (1.0 / (k_trunc - k_lead))


def _smallest_subnormal(dtype):
    """Smallest positive subnormal float in `dtype`."""
    info = np.finfo(dtype)
    if hasattr(info, "smallest_subnormal"):
        return float(info.smallest_subnormal)
    # numpy < 1.22 fallback
    return math.ldexp(1.0, info.minexp - info.nmant)


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
    smallest_subnormal = _smallest_subnormal(dtype)

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


def closed_branch_rel_err_bound(cache, n, v, order, *, dtype=None):
    """Upper bound on the closed branch's relative error at |x-a| = v.

    Derivation. Under the pristine-f contract (f's evaluation has
    relative error ≤ ε_m), the closed branch's numerator absolute
    error is bounded by

        |abs_err_numerator| ≤ ε_m · Σ_{i ∈ [0, n)} |c_i · v^i|       (Wilkinson, leading term)

    The numerator's true magnitude at |x-a|=v is |c_{k_lead}| · v^{k_lead}
    where k_lead is the smallest k ≥ n with c_k ≠ 0. Dividing by v^n
    preserves relative error (one extra rounding, absorbed into ε_m), so

        rel_err_closed(v)  ≤  ε_m · Σ|c_i·v^i| / (|c_{k_lead}| · v^{k_lead}).

    Two regimes:
      - k_lead == n (typical): the formula reduces to the single-term
        cancellation model, with closed_branch_rel_err_bound bounded
        below tol_rel at the auto_eps boundary by construction.
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
    # Cancellation contribution: ε_m · Σ|c_i·v^i| / (|c_kl|·v^kl).
    # Plus a 1·ε_m floor: even with zero cancellation, the final
    # division (numerator / v^n) introduces one ULP of rounding error.
    cancellation = combined / (v_lead * abs(v) ** k_lead) if combined > 0 else 0.0
    return eps_machine * (cancellation + 1.0)


def poly_branch_rel_err_bound(order, *, dtype=None):
    """Upper bound on the polynomial branch's relative error from
    floating-point evaluation: (order+1)·ε_m by Wilkinson's bound on
    Horner-method summation of `order` terms (each multiplication and
    addition contributes one ULP).

    Holds when truncation error is sub-dominant (i.e., |x-a| < eps where
    eps was chosen by auto_eps to drive truncation below tol_rel).
    """
    if dtype is None:
        dtype = np.float64
    eps_machine = float(np.finfo(np.dtype(dtype)).eps)
    return (order + 1) * eps_machine


def check_closed_cancellation_safety(cache, n, eps, order, *, dtype=None, max_extra=4):
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
    tol_rel = 10.0 * eps_machine

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


def closed_branch_needed(cache, n, order, t_max, *, dtype=None, max_extra=4):
    """Return True if the polynomial branch alone is insufficient over |t| <= t_max.

    When False, you can drop the closed branch entirely -- polynomial achieves
    ~10·eps_machine relative accuracy throughout the input range, saving a
    transcendental call per element.
    """
    if dtype is None:
        dtype = np.dtype(cache.x.dtype)
    else:
        dtype = np.dtype(dtype)
    eps_machine = float(np.finfo(dtype).eps)
    tol_rel = 10.0 * eps_machine

    v_lead, k_lead = _first_nonvanishing(cache, n, order)
    if v_lead == 0.0:
        return False
    v_trunc, k_trunc = _first_nonvanishing(cache, n + order, max_extra + 1)
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

    poly = coeffs[n]
    for k in range(1, order):
        poly = poly + coeffs[n + k] * t**k

    if n == 0:
        closed = f
    else:
        P = coeffs[0]
        for k in range(1, n):
            P = P + coeffs[k] * t**k
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
