"""Adversarial parameter scan for taylor_remainder at n=3.

n=3 layers two new structural challenges on top of n=2:

  1. Three-piece P_2 = c_0 + c_1·x + c_2·x²: canonicalize must fold three
     duplicate sub-trees in (f - P_2), not one (n=1) or two (n=2).
  2. Division by x³: residual numerator error is amplified by 1/|x|³,
     so any leak from incomplete cancellation grows fast as x -> 0.

This suite hammers the four canonical sparsity patterns the cache may
encounter:

  - dense odd-only (sin(K·x)):     k_lead = 3, gap = 12
  - dense even-only (cos(K·x)):    k_lead = 4 (c_3 vanishes), gap = 10
  - parity-4 (cos(K·x²)):          k_lead = 4, k_trunc = 16, gap = 12
                                   (eps scales as 1/sqrt(K), not 1/K)
  - dense (K3·x³ · cos(L·x)):      R = K3·cos(L·x) exactly, no numerator
                                   cancellation -- a clean reference

Each is paired with a coefficient-magnitude scan: we layer K0, K1, K2 on
top of a base f to manufacture a rich P_2 to be folded, sweeping each
across at least 30 orders of magnitude.

References computed at 60 digits with mpmath.
"""

import math
import warnings

import mpmath as mp
import numpy as np
import pytest
from taylor_remainder import (
    TaylorAtPoint,
    TaylorRemainderClosedCancellationWarning,
    auto_eps,
    closed_branch_rel_err_bound,
    poly_branch_rel_err_bound,
    taylor_remainder,
    taylor_remainder_poly,
)

import pytensor
import pytensor.tensor as pt


mp.mp.dps = 60
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.on_opt_error = "ignore"


# ---- mpmath references -----------------------------------------------------


def _R_sin_Kx(K, x_mp):
    """f = sin(K·x), n=3, a=0:  (sin(Kx) - K·x) / x³.  R(0) = -K³/6."""
    if x_mp == 0:
        return -(mp.mpf(K) ** 3) / 6
    Kx = mp.mpf(K) * x_mp
    return (mp.sin(Kx) - Kx) / x_mp**3


def _R_cos_Kx(K, x_mp):
    """f = cos(K·x), n=3, a=0:  (cos(Kx) - 1 + K²x²/2) / x³.  R(0) = 0."""
    if x_mp == 0:
        return mp.mpf(0)
    Kx = mp.mpf(K) * x_mp
    return (mp.cos(Kx) - 1 + Kx**2 / 2) / x_mp**3


def _R_polynomial_parity4(a, b, c, d, x_v):
    """f = 1 + a·x⁴ + b·x⁸ + c·x^12 + d·x^16, n=3:
    R(x) = (f - 1)/x³ = a·x + b·x⁵ + c·x⁹ + d·x^13."""
    return a * x_v + b * x_v**5 + c * x_v**9 + d * x_v**13


def _R_x3_cos_Lx(K3, L, x_mp):
    """f = K3·x³ · cos(L·x), n=3:  R(x) = K3·cos(L·x) exactly."""
    return mp.mpf(K3) * mp.cos(mp.mpf(L) * x_mp)


def _R_layered(K0, K1, K2, K3, L, x_mp):
    """f = K0 + K1·x + K2·x² + K3·x³·cos(L·x), n=3:  R = K3·cos(Lx) since
    P_2 = K0 + K1·x + K2·x² folds the layered terms exactly."""
    return mp.mpf(K3) * mp.cos(mp.mpf(L) * x_mp)


# ---- forward accuracy: dense odd-only (sin(K·x)) ---------------------------


@pytest.mark.parametrize("K", [1e-3, 0.1, 1.0, 10.0, 100.0, 1000.0])
@pytest.mark.parametrize("order", [10, 14])
def test_n3_sin_Kx_forward(K, order):
    """f = sin(K·x), n=3: c_2 = 0, c_3 = -K³/6 leading. Closed branch
    must fold (P_2 = K·x) cancellation when K is large."""
    x = pt.dscalar("x")
    f = pt.sin(K * x)
    cache = TaylorAtPoint(f, x, 0.0)
    eps = auto_eps(cache, n=3, order=order)
    y = taylor_remainder(f, x, 0.0, 3, order=order)
    fn = pytensor.function([x], y)

    fracs = (1e-6, 0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 5.0)
    sweep = [0.0, *(eps * fr for fr in fracs if eps * fr <= 0.5)]
    for v in sweep:
        ref = float(_R_sin_Kx(K, mp.mpf(v)))
        out = float(fn(v))
        assert math.isfinite(out), f"K={K} x={v}: not finite"
        ref_safe = abs(ref) if abs(ref) > 1e-300 else max(1.0, abs(K) ** 3 / 6)
        rel = abs(out - ref) / ref_safe
        # Tolerance: poly bound for v inside window, closed-branch
        # cancellation bound for v outside.  No magic constants.
        tol = (
            poly_branch_rel_err_bound(order=order, cache=cache, n=3, v=v)
            if v < eps
            else closed_branch_rel_err_bound(cache, n=3, v=v, order=order)
        )
        assert rel <= tol, (
            f"K={K} order={order} x={v} (eps={eps:.3g}): "
            f"got {out}, ref {ref}, rel_err {rel:.2e} > tol {tol:.2e}"
        )


# ---- forward accuracy: dense even-only (cos(K·x)), c_3 = 0 -----------------


@pytest.mark.parametrize("K", [0.1, 1.0, 10.0, 100.0])
def test_n3_cos_Kx_forward(K):
    """f = cos(K·x), n=3: c_3 = 0 so k_lead = 4 -- the numerator
    (cos(Kx) - 1 + K²x²/2) has multi-term cancellation (both c_0 = 1
    and c_2·v² = (K²/2)·v² fold against f's contributions).

    The tolerance comes directly from the same first-principles error
    bound that taylor_remainder's check_closed_cancellation_safety uses
    to predict the boundary error and warn the user:

        rel_err_closed(v)  ≤  ε_m · Σ_{i ∈ [0, n)} |c_i · v^i|
                                  / (|c_{k_lead}| · v^{k_lead}).

    Inside the poly window (|v| < eps): poly-evaluation rounding bound
    is (order+1)·ε_m by Wilkinson on Horner. No magic constants,
    no safety multipliers -- the test passes whenever the realized error
    stays under the bound the warning is computed from.
    """
    x = pt.dscalar("x")
    f = pt.cos(K * x)
    cache = TaylorAtPoint(f, x, 0.0)
    # The closed-cancellation warning fires at this k_lead > n
    # configuration; the test's job is to verify the observed error
    # stays within the bound the warning is derived from.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TaylorRemainderClosedCancellationWarning)
        eps = auto_eps(cache, n=3, order=10)
        y = taylor_remainder(f, x, 0.0, 3, order=10)
    fn = pytensor.function([x], y)

    fracs = (1e-6, 0.1, 0.5, 0.9, 0.99, 1.0, 1.001, 1.01, 1.1, 1.5, 2.0, 5.0)
    sweep = [0.0, *(eps * fr for fr in fracs if eps * fr <= 0.5)]
    for v in sweep:
        ref = float(_R_cos_Kx(K, mp.mpf(v)))
        out = float(fn(v))
        assert math.isfinite(out), f"K={K} x={v}: not finite"
        # R(0) = 0 has no natural magnitude scale; use K³ as proxy
        # (R(eps) ~ K⁵·eps²/120 ~ K³ at boundary for the next-order term).
        ref_safe = abs(ref) if abs(ref) > 1e-300 else max(1.0, abs(K) ** 3)
        rel = abs(out - ref) / ref_safe
        # Tolerance: poly bound for v inside window, closed-branch
        # cancellation bound for v outside.  Switch is at v == eps.
        if v < eps:
            tol = poly_branch_rel_err_bound(order=10, cache=cache, n=3, v=v)
        else:
            tol = closed_branch_rel_err_bound(cache, n=3, v=v, order=10)
        assert rel <= tol, (
            f"K={K} x={v} (eps={eps:.3g}): got {out}, ref {ref}, "
            f"rel_err {rel:.2e} > tol {tol:.2e}"
        )


# ---- forward accuracy: parity-4 sparsity (synthetic polynomial) ------------


@pytest.mark.parametrize(
    "a,b,c,d",
    [
        (1.0, 1.0, 1.0, 1.0),  # all coefficients unit
        (1e10, 1.0, 1.0, 1.0),  # leading c_4 huge
        (1.0, 1.0, 1.0, 1e-20),  # k_trunc coefficient tiny
        (-1.0, 1.0, -1.0, 1.0),  # alternating signs
    ],
)
def test_n3_polynomial_parity4_sparse(a, b, c, d):
    """f = 1 + a·x⁴ + b·x⁸ + c·x^12 + d·x^16, n=3.

    Parity-4 sparsity: only c_{4j} nonzero, with c_3 = 0. auto_eps must
    detect k_lead = 4 and k_trunc = 16 (gap = 12).  R(x) = a·x + b·x⁵ +
    c·x⁹ + d·x^13 exactly, regardless of branch.

    Polynomial f sidesteps the deep-grad chain that derails cos(K·x²) at
    higher orders -- pt.grad on a polynomial is linear and fast -- so
    this is a clean test of auto_eps's parity-4 path without fighting
    pytensor's grad recursion limits.
    """
    x = pt.dscalar("x")
    f = 1.0 + a * x**4 + b * x**8 + c * x**12 + d * x**16

    cache = TaylorAtPoint(f, x, 0.0)
    # Confirm the parity-4 structure is detected.
    assert cache.numeric_coeff(3) == 0.0
    assert cache.numeric_coeff(4) != 0.0
    assert cache.numeric_coeff(5) == 0.0
    assert cache.numeric_coeff(8) != 0.0
    assert cache.numeric_coeff(16) != 0.0

    eps = auto_eps(cache, n=3, order=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TaylorRemainderClosedCancellationWarning)
        y = taylor_remainder(f, x, 0.0, 3, order=10)
    fn = pytensor.function([x], y)

    sweep = [0.0, 1e-12, 1e-8, 1e-4, 1e-2, 0.1, 0.3, 0.5]
    for v in sweep:
        ref = _R_polynomial_parity4(a, b, c, d, v)
        out = float(fn(v))
        assert math.isfinite(out), f"a={a} b={b} c={c} d={d} x={v}: not finite"
        ref_safe = abs(ref) if abs(ref) > 1e-300 else max(1.0, abs(a))
        rel = abs(out - ref) / ref_safe
        # The closed_branch_rel_err_bound is a CONSERVATIVE upper bound
        # assuming the worst-case multi-term cancellation. For polynomial
        # f, canonicalize folds (f - P_2) symbolically, so the actual
        # closed branch evaluates a clean polynomial residual with no
        # cancellation -- the bound holds with lots of slack.
        tol = (
            poly_branch_rel_err_bound(order=10, cache=cache, n=3, v=v)
            if v < eps
            else closed_branch_rel_err_bound(cache, n=3, v=v, order=10)
        )
        assert rel <= tol, (
            f"a={a} b={b} c={c} d={d} x={v}: got {out}, ref {ref}, "
            f"rel_err {rel:.2e} > tol {tol:.2e}"
        )


# ---- parity-4 sparsity via cos(K·x²) using mpmath-computed coefficients ----


def _populate_cache_via_mpmath(cache, f_mp_callable, a, n_coeffs):
    """Populate `cache` with Taylor coefficients of `f_mp_callable` at
    a, using mpmath. Threshold below dtype eps_machine·max(|c_i|) to
    silence numerical-differentiation noise that would otherwise be
    picked up by _first_nonvanishing in auto_eps."""
    coeffs_mp = mp.taylor(f_mp_callable, a, n_coeffs - 1)
    coeffs = [float(c) for c in coeffs_mp]
    threshold = float(np.finfo(np.float64).eps) * max(abs(c) for c in coeffs)
    coeffs = [0.0 if abs(c) < threshold else c for c in coeffs]
    cache.populate_from_coefficients(coeffs)


@pytest.mark.parametrize("K", [0.1, 1.0, 10.0])
def test_n3_cos_Kxsq_parity4_via_mpmath(K):
    """f = cos(K·x²), n=3: parity-4 sparsity (k_lead=4, k_trunc=16,
    gap=12), eps scales as 1/sqrt(K).

    cos(K·x²) is mathematically well-behaved but its symbolic grad
    chain in pytensor blows up exponentially -- by m=14 a single
    derivative substitution takes minutes. We bypass that by computing
    coefficients with mpmath and pre-populating the cache; the closed
    branch still uses pt.cos(K*x**2) at runtime. The test exercises
    the same principled tolerance pattern as the other n=3 tests.
    """
    x = pt.dscalar("x")
    f = pt.cos(K * x**2)

    def f_mp(y, K=K):
        return mp.cos(K * y**2)

    cache = TaylorAtPoint(f, x, 0.0)
    # Need coefficients up to n + order + max_extra - 1 = 16
    _populate_cache_via_mpmath(cache, f_mp, 0.0, n_coeffs=18)
    assert cache.numeric_coeff(3) == 0.0  # parity-4
    assert cache.numeric_coeff(4) != 0.0
    assert cache.numeric_coeff(15) == 0.0  # noise filtered
    assert cache.numeric_coeff(16) != 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TaylorRemainderClosedCancellationWarning)
        eps = auto_eps(cache, n=3, order=10)
        y = taylor_remainder(f, x, 0.0, 3, order=10, cache=cache)
    fn = pytensor.function([x], y)

    def _R_cos_Kxsq_high_precision(K, v):
        """Reference R(v) = (cos(K·v²) - 1)/v³ at very high precision.
        At small v, cos(K·v²) - 1 has catastrophic cancellation that
        mpmath at 60 dps cannot resolve (the leading -K²·v⁴/2 term is
        ~v⁴ smaller than cos's value 1, so 60 dps loses ~50 digits).
        We bump to 200 dps locally to recover the true mathematical value."""
        if v == 0:
            return 0.0
        with mp.workdps(200):
            mv = mp.mpf(v)
            return float((mp.cos(K * mv**2) - 1) / mv**3)

    sweep = [0.0, 1e-12, 1e-6, 1e-3, 0.01, 0.05, 0.1, 0.3]
    for v in sweep:
        ref = _R_cos_Kxsq_high_precision(K, v)
        out = float(fn(v))
        assert math.isfinite(out), f"K={K} x={v}: not finite"
        ref_safe = abs(ref) if abs(ref) > 1e-300 else max(1.0, abs(K) ** 2)
        rel = abs(out - ref) / ref_safe
        tol = (
            poly_branch_rel_err_bound(order=10, cache=cache, n=3, v=v)
            if v < eps
            else closed_branch_rel_err_bound(cache, n=3, v=v, order=10)
        )
        assert rel <= tol, (
            f"K={K} x={v} (eps={eps:.3g}): "
            f"got {out}, ref {ref}, rel_err {rel:.2e} > tol {tol:.2e}"
        )


# ---- coefficient-magnitude scan: K0/K1/K2 layered onto K3·x³·cos(L·x) -----


@pytest.mark.parametrize("K0_log10", [-30, 0, 10])
@pytest.mark.parametrize("K1_log10", [-30, 0, 10])
@pytest.mark.parametrize("K2_log10", [-30, 0, 10])
def test_n3_layered_three_subtree_fold(K0_log10, K1_log10, K2_log10):
    """f = K0 + K1·x + K2·x² + x³·cos(x), n=3.

    The point is to manufacture a non-trivial P_2 = K0 + K1·x + K2·x²
    that canonicalize must fold THREE separate sub-trees of (f - P_2):
    a constant, a linear term, and a quadratic term. The closed branch
    is correct only if all three fold cleanly. K3 = 1 fixed; L = 1.

    True R = cos(x) regardless of K0/K1/K2 (those cancel exactly in
    f - P_2). So the test is "does the symbolic cancellation actually
    happen at runtime?" across many magnitude combinations.
    """
    K0 = 10.0**K0_log10
    K1 = 10.0**K1_log10
    K2 = 10.0**K2_log10

    x = pt.dscalar("x")
    f = K0 + K1 * x + K2 * x**2 + x**3 * pt.cos(x)
    cache = TaylorAtPoint(f, x, 0.0)
    eps = auto_eps(cache, n=3, order=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TaylorRemainderClosedCancellationWarning)
        y = taylor_remainder(f, x, 0.0, 3, order=10, cache=cache)
    fn = pytensor.function([x], y)

    # Span small (poly-branch dominant) to moderate (closed-branch dominant)
    sweep = [0.0, 1e-12, 1e-6, 1e-3, 0.01, 0.1, 0.3]
    for v in sweep:
        ref = float(_R_layered(K0, K1, K2, 1.0, 1.0, mp.mpf(v)))
        out = float(fn(v))
        assert math.isfinite(out), f"K0={K0} K1={K1} K2={K2} x={v}: not finite ({out})"
        ref_safe = abs(ref) if abs(ref) > 1e-300 else 1.0
        rel = abs(out - ref) / ref_safe
        tol = (
            poly_branch_rel_err_bound(order=10, cache=cache, n=3, v=v)
            if v < eps
            else closed_branch_rel_err_bound(cache, n=3, v=v, order=10)
        )
        assert rel <= tol, (
            f"K0={K0} K1={K1} K2={K2} x={v}: "
            f"got {out}, ref {ref}, rel_err {rel:.2e} > tol {tol:.2e}"
        )


# ---- K3 scale invariance ---------------------------------------------------


@pytest.mark.parametrize("K3_log10", [-50, -10, 0, 10, 50])
def test_n3_K3_scale_invariant_sin(K3_log10):
    """f = K3 · sin(x), n=3: R(x) = K3·(sin(x) - x)/x³, R(0) = -K3/6.
    Multiplying f by a constant should leave the *relative* accuracy of
    R unchanged -- auto_eps is scale-invariant in f."""
    K3 = 10.0**K3_log10
    x = pt.dscalar("x")
    f = K3 * pt.sin(x)
    cache = TaylorAtPoint(f, x, 0.0)
    eps = auto_eps(cache, n=3, order=10)
    # Scale-invariance: eps should match auto_eps(sin(x))
    eps_unit = auto_eps(TaylorAtPoint(pt.sin(x), x, 0.0), n=3, order=10)
    assert math.isclose(eps, eps_unit, rel_tol=1e-12), (
        f"K3={K3}: eps={eps} != eps_unit={eps_unit}"
    )

    y = taylor_remainder(f, x, 0.0, 3, order=10)
    fn = pytensor.function([x], y)

    for v in [0.0, 1e-10, 1e-5, 0.01, 0.1, 0.3]:
        ref = float(_R_sin_Kx(1.0, mp.mpf(v))) * K3
        out = float(fn(v))
        assert math.isfinite(out), f"K3={K3} x={v}: not finite ({out})"
        ref_safe = abs(ref) if abs(ref) > 1e-300 else max(K3, 1.0) / 6
        rel = abs(out - ref) / ref_safe
        tol = (
            poly_branch_rel_err_bound(order=10, cache=cache, n=3, v=v)
            if v < eps
            else closed_branch_rel_err_bound(cache, n=3, v=v, order=10)
        )
        assert rel <= tol, f"K3={K3} x={v}: got {out}, ref {ref}, rel_err {rel:.2e}"


# ---- inner-frequency scaling for cos(K·x) at n=3 ---------------------------


def test_n3_K_times_eps_invariant_cos():
    """For f = cos(K·x), n=3: K·auto_eps should be invariant in K. This
    is the natural eps in y = K·x where Taylor coefficients of cos(y)
    don't depend on K. Same property held at n=2; we re-test at n=3
    because k_lead has shifted from 2 to 4."""
    x = pt.dscalar("x")
    pairs = []
    for K in [1e-3, 1.0, 10.0, 100.0, 1000.0]:
        f = pt.cos(K * x)
        cache = TaylorAtPoint(f, x, 0.0)
        eps = auto_eps(cache, n=3, order=10)
        pairs.append((K, K * eps))
    refs = [v for _, v in pairs]
    rel_spread = (max(refs) - min(refs)) / refs[0]
    assert rel_spread < 1e-12, f"K·eps not invariant at n=3: {pairs}"


# ---- iterated grad of R = K3·cos(L·x) at x = 0 -----------------------------
#
# For f = x³·cos(L·x), R(x) = cos(L·x).  At x = 0:
#   R^(2j)(0) = (-1)^j · L^(2j),   R^(2j+1)(0) = 0.


def _ref_grad_cos_Lx(L, k):
    if k % 2 == 1:
        return 0.0
    j = k // 2
    return (-1) ** j * L ** (2 * j)


@pytest.mark.parametrize("L", [0.5, 1.0, 3.0])
def test_n3_iterated_grad_x3_cos_Lx_poly(L):
    """f = x³·cos(L·x), n=3. R = cos(L·x) symbolically, so iterated grad
    at x = 0 just walks the cos(L·x) Taylor series.

    Use the polynomial-only variant: full taylor_remainder's closed
    branch is x³·cos(Lx)/x³ which is mathematically cos(Lx), but the
    grad of (... )/x³ blows up at x = 0 in float, even after canonicalize
    might fold things in FAST_RUN. The poly variant has no closed branch.
    """
    x = pt.dscalar("x")
    f = x**3 * pt.cos(L * x)
    cur = taylor_remainder_poly(f, x, 0.0, 3, order=14)

    tol = 1e-10
    for k in range(6):
        out = float(pytensor.function([x], cur)(0.0))
        ref = _ref_grad_cos_Lx(L, k)
        ref_safe = abs(ref) if abs(ref) > 1e-300 else 1.0
        rel = abs(out - ref) / ref_safe
        assert rel <= tol, f"L={L} k={k}: got {out}, ref {ref}, rel_err {rel:.2e}"
        if k < 5:
            cur = pt.grad(cur, x)


# ---- switch-boundary continuity --------------------------------------------


@pytest.mark.parametrize("K", [1.0, 10.0, 100.0])
def test_n3_switch_boundary_continuity_sin(K):
    """At |x| = eps both branches must agree.  n=3 is more sensitive
    than n=2 because errors get divided by x³ instead of x²."""
    x = pt.dscalar("x")
    f = pt.sin(K * x)
    cache = TaylorAtPoint(f, x, 0.0)
    eps = auto_eps(cache, n=3, order=10)
    y = taylor_remainder(f, x, 0.0, 3, order=10, eps=eps)
    fn = pytensor.function([x], y)

    for frac in (0.999, 0.9999, 1.0001, 1.001):
        v = eps * frac
        ref = float(_R_sin_Kx(K, mp.mpf(v)))
        out = float(fn(v))
        ref_safe = abs(ref) if abs(ref) > 1e-300 else max(1.0, abs(K) ** 3 / 6)
        rel = abs(out - ref) / ref_safe
        tol = (
            poly_branch_rel_err_bound(order=10, cache=cache, n=3, v=v)
            if v < eps
            else closed_branch_rel_err_bound(cache, n=3, v=v, order=10)
        )
        assert rel <= tol, (
            f"K={K} frac={frac} v={v}: got {out}, ref {ref}, rel_err {rel:.2e}"
        )


# ---- pristine vs. layered: same R, different f shapes ----------------------


@pytest.mark.parametrize("K0_log10", [0, 10])
@pytest.mark.parametrize("K1_log10", [0, 10])
@pytest.mark.parametrize("K2_log10", [0, 10])
def test_n3_layered_matches_pristine_x3_cos(K0_log10, K1_log10, K2_log10):
    """f_pristine = x³·cos(x) and f_layered = K0 + K1·x + K2·x² + x³·cos(x)
    should give identical R = cos(x) up to ~tol after canonicalize folds
    the K0/K1·x/K2·x² terms.  Direct parity check across coefficient
    magnitudes."""
    K0 = 10.0**K0_log10
    K1 = 10.0**K1_log10
    K2 = 10.0**K2_log10

    x = pt.dscalar("x")
    f_pristine = x**3 * pt.cos(x)
    f_layered = K0 + K1 * x + K2 * x**2 + x**3 * pt.cos(x)

    cache_p = TaylorAtPoint(f_pristine, x, 0.0)
    cache_l = TaylorAtPoint(f_layered, x, 0.0)
    eps_p = auto_eps(cache_p, n=3, order=10)
    eps_l = auto_eps(cache_l, n=3, order=10)
    fn_p = pytensor.function(
        [x], taylor_remainder(f_pristine, x, 0.0, 3, order=10, cache=cache_p)
    )
    fn_l = pytensor.function(
        [x], taylor_remainder(f_layered, x, 0.0, 3, order=10, cache=cache_l)
    )

    for v in [0.0, 1e-12, 1e-6, 1e-3, 0.01, 0.1, 0.3]:
        out_p = float(fn_p(v))
        out_l = float(fn_l(v))
        ref = math.cos(v)
        rel_p = abs(out_p - ref) / max(1.0, abs(ref))
        rel_l = abs(out_l - ref) / max(1.0, abs(ref))
        tol_p = (
            poly_branch_rel_err_bound(order=10, cache=cache_p, n=3, v=v)
            if v < eps_p
            else closed_branch_rel_err_bound(cache_p, n=3, v=v, order=10)
        )
        tol_l = (
            poly_branch_rel_err_bound(order=10, cache=cache_l, n=3, v=v)
            if v < eps_l
            else closed_branch_rel_err_bound(cache_l, n=3, v=v, order=10)
        )
        assert rel_p <= tol_p, (
            f"K0={K0} K1={K1} K2={K2} x={v}: pristine rel_err {rel_p:.2e} > tol {tol_p:.2e}"
        )
        assert rel_l <= tol_l, (
            f"K0={K0} K1={K1} K2={K2} x={v}: layered rel_err {rel_l:.2e} > tol {tol_l:.2e}"
        )
