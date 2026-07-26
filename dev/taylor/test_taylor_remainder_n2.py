"""Adversarial parameter scan for taylor_remainder at n=2.

The n=2 case has its own structural challenges that n=1 does not:

  - The closed branch subtracts a degree-1 polynomial P_1(x) = c_0 + c_1·x,
    so canonicalize must fold *two* duplicate sub-trees (the constant K0
    and the linear K1·x term), not just one.
  - The denominator is x², so any residual numerator error is amplified
    by 1/x², making the n=2 closed branch more sensitive than n=1.
  - Functions like cos(K·x) have a sparse Taylor series (only even powers),
    pushing the polynomial-truncation gap to k_trunc - k_lead = 10 even
    at order=10. eps is then natural in the units of the function's
    intrinsic frequency: eps ∝ 1/K.

This suite hammers cos(K·x) and family across many K values, plus the
sparse-leading case (sin(K·x) at n=2 has c_2 = 0, so k_lead jumps to 3).
References computed to 60 digits with mpmath.
"""

import math

import mpmath as mp
import numpy as np
import pytest
from taylor_remainder import (
    TaylorAtPoint,
    auto_eps,
    taylor_remainder,
    taylor_remainder_poly,
)

import pytensor
import pytensor.tensor as pt


mp.mp.dps = 60
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.on_opt_error = "ignore"


def _R_cos_Kx(K, x_mp):
    """R(x) for f = cos(K·x), n=2, a=0:  (cos(Kx) - 1) / x²."""
    if x_mp == 0:
        return -(mp.mpf(K) ** 2) / 2
    Kx = mp.mpf(K) * x_mp
    return (mp.cos(Kx) - 1) / x_mp**2


def _R_outer_cos(K0, K_outer, K_inner, x_mp):
    """R(x) for f = K0 + K_outer · (cos(K_inner·x) - 1), n=2:
    K_outer · (cos(K_inner x) - 1) / x²."""
    if x_mp == 0:
        return -mp.mpf(K_outer) * mp.mpf(K_inner) ** 2 / 2
    Kx = mp.mpf(K_inner) * x_mp
    return mp.mpf(K_outer) * (mp.cos(Kx) - 1) / x_mp**2


def _R_polynomial_P1(K0, K1, K_outer, K_inner, x_mp):
    """R(x) for f = K0 + K1·x + K_outer·(cos(K_inner·x) - 1), n=2:
    canonicalize must fold both the K0 cancellation AND the K1·x term."""
    if x_mp == 0:
        return -mp.mpf(K_outer) * mp.mpf(K_inner) ** 2 / 2
    Kx = mp.mpf(K_inner) * x_mp
    return mp.mpf(K_outer) * (mp.cos(Kx) - 1) / x_mp**2


def _R_sin_Kx(K, x_mp):
    """R(x) for f = sin(K·x), n=2:  c_2 = 0, leading is c_3 = -K³/6.
    R(x) = (sin(Kx) - K·x) / x²."""
    if x_mp == 0:
        return mp.mpf(0)
    Kx = mp.mpf(K) * x_mp
    return (mp.sin(Kx) - Kx) / x_mp**2


# ---- forward accuracy: cos(K·x) sweep --------------------------------------


@pytest.mark.parametrize("K", [1e-3, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0])
@pytest.mark.parametrize("order", [10])
def test_n2_cos_Kx_forward(K, order):
    """f = cos(K·x), n=2: matches (cos(Kx)-1)/x² to ~10·eps_machine across
    a sweep that straddles auto_eps."""
    x = pt.dscalar("x")
    f = pt.cos(K * x)
    cache = TaylorAtPoint(f, x, 0.0)
    eps = auto_eps(cache, n=2, order=order)
    y = taylor_remainder(f, x, 0.0, 2, order=order)
    fn = pytensor.function([x], y)

    # Sweep includes points well inside, near, and well outside the eps
    # boundary, plus exact x = ±eps.
    fracs = (1e-8, 1e-3, 0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 5.0)
    # Cap |x| at 0.5 -- beyond that cos(Kx) for large K saturates and the
    # test loses physical meaning.
    sweep = [0.0, *(eps * f for f in fracs if eps * f <= 0.5)]

    tol = 100 * np.finfo(np.float64).eps
    for v in sweep:
        ref = float(_R_cos_Kx(K, mp.mpf(v)))
        out = float(fn(v))
        assert math.isfinite(out), f"K={K} x={v}: not finite ({out})"
        ref_safe = abs(ref) if abs(ref) > 1e-300 else 1.0
        rel = abs(out - ref) / ref_safe
        assert rel <= tol, (
            f"K={K} order={order} x={v} (eps={eps:.3g}): "
            f"got {out}, ref {ref}, rel_err {rel:.2e} > tol {tol:.2e}"
        )


# ---- forward accuracy: K0 + K_outer · (cos(K_inner·x) - 1) -----------------


@pytest.mark.parametrize("K_inner", [1.0, 10.0, 100.0])
@pytest.mark.parametrize("K_outer_log10", [-30, 0, 30])
@pytest.mark.parametrize("K0_choice", ["zero", "1", "1e10"])
def test_n2_outer_cos_forward(K_inner, K_outer_log10, K0_choice):
    """f = K0 + K_outer · (cos(K_inner·x) - 1), n=2. The K0 cancellation
    is the same as the n=1 K0 case but with x² in the denominator, so
    any residual K0 mismatch in the closed branch gets amplified."""
    K_outer = 10.0**K_outer_log10
    K0 = {"zero": 0.0, "1": 1.0, "1e10": 1e10}[K0_choice]

    x = pt.dscalar("x")
    f = K0 + K_outer * (pt.cos(K_inner * x) - 1)
    y = taylor_remainder(f, x, 0.0, 2, order=10)
    fn = pytensor.function([x], y)

    sweep = [0.0, 1e-12, 1e-8, 1e-4, 1e-2, 0.1, 0.3]
    tol = max(1e-12, 100 * np.finfo(np.float64).eps)
    for v in sweep:
        ref = float(_R_outer_cos(K0, K_outer, K_inner, mp.mpf(v)))
        out = float(fn(v))
        assert math.isfinite(out), (
            f"K_inner={K_inner} K_outer={K_outer} K0={K0} x={v}: not finite"
        )
        ref_safe = abs(ref) if abs(ref) > 1e-300 else 1.0
        rel = abs(out - ref) / ref_safe
        assert rel <= tol, (
            f"K_inner={K_inner} K_outer={K_outer} K0={K0} x={v}: "
            f"got {out}, ref {ref}, rel_err {rel:.2e} > tol {tol:.2e}"
        )


# ---- forward accuracy: P_1 with explicit linear term -----------------------


@pytest.mark.parametrize("K1_log10", [-10, 0, 10])
@pytest.mark.parametrize("K0_choice", ["zero", "1e5"])
def test_n2_explicit_P1_with_linear_term(K1_log10, K0_choice):
    """f = K0 + K1·x + (cos(x) - 1), n=2. P_1 = K0 + K1·x: canonicalize
    must fold BOTH the K0 cancellation and the K1·x term to keep the
    closed branch accurate."""
    K1 = 10.0**K1_log10
    K0 = {"zero": 0.0, "1e5": 1e5}[K0_choice]

    x = pt.dscalar("x")
    f = K0 + K1 * x + (pt.cos(x) - 1)
    y = taylor_remainder(f, x, 0.0, 2, order=10)
    fn = pytensor.function([x], y)

    sweep = [0.0, 1e-12, 1e-8, 1e-4, 1e-2, 0.1, 0.3]
    tol = max(1e-12, 100 * np.finfo(np.float64).eps)
    for v in sweep:
        ref = float(_R_polynomial_P1(K0, K1, 1.0, 1.0, mp.mpf(v)))
        out = float(fn(v))
        assert math.isfinite(out), f"K0={K0} K1={K1} x={v}: not finite"
        ref_safe = abs(ref) if abs(ref) > 1e-300 else 1.0
        rel = abs(out - ref) / ref_safe
        assert rel <= tol, (
            f"K0={K0} K1={K1} x={v}: got {out}, ref {ref}, rel_err {rel:.2e} > tol {tol:.2e}"
        )


# ---- forward accuracy: sparse-leading c_2 = 0 (sin family) -----------------


@pytest.mark.parametrize("K", [0.1, 1.0, 10.0, 100.0])
def test_n2_sin_Kx_sparse_leading(K):
    """f = sin(K·x), n=2: c_2 = 0, so auto_eps must detect the leading
    coefficient at k_lead = 3 (c_3 = -K³/6) and pick eps based on the
    k=3 -> k=11 gap, not the k=2 -> k=12 gap."""
    x = pt.dscalar("x")
    f = pt.sin(K * x)

    cache = TaylorAtPoint(f, x, 0.0)
    # Confirm c_2 is symbolically zero (sparse-leading detection works).
    assert cache.numeric_coeff(2) == 0.0
    assert cache.numeric_coeff(3) != 0.0

    y = taylor_remainder(f, x, 0.0, 2, order=10)
    fn = pytensor.function([x], y)

    sweep = [0.0, 1e-10, 1e-5, 1e-2, 0.1, 0.3]
    tol = max(1e-12, 100 * np.finfo(np.float64).eps)
    for v in sweep:
        ref = float(_R_sin_Kx(K, mp.mpf(v)))
        out = float(fn(v))
        assert math.isfinite(out), f"K={K} x={v}: not finite"
        ref_safe = abs(ref) if abs(ref) > 1e-300 else 1.0
        rel = abs(out - ref) / ref_safe
        assert rel <= tol, (
            f"K={K} x={v}: got {out}, ref {ref}, rel_err {rel:.2e} > tol {tol:.2e}"
        )


# ---- iterated grad of R(x) for f = cos(K·x) at x = 0 -----------------------
#
# R(x) = (cos(Kx) - 1) / x² = -K²/2 + K⁴·x²/24 - K⁶·x⁴/720 + ...
#      = Σ_{j>=0} (-1)^(j+1) K^(2j+2) x^(2j) / (2j+2)!
#
# So R^(2j)(0) = (2j)! · (-1)^(j+1) K^(2j+2) / (2j+2)!
#             = (-1)^(j+1) K^(2j+2) / ((2j+1)(2j+2))
# and R^(2j+1)(0) = 0.


def _ref_grad_at_0_cos_Kx(K, k):
    """k-th derivative of R = (cos(Kx) - 1)/x² at x = 0."""
    if k % 2 == 1:
        return 0.0
    j = k // 2
    return (-1) ** (j + 1) * K ** (2 * j + 2) / ((2 * j + 1) * (2 * j + 2))


@pytest.mark.parametrize("K", [0.5, 1.0, 3.0, 10.0])
def test_n2_iterated_grad_cos_Kx_poly(K):
    """Take 0..5 derivatives of R = (cos(Kx) - 1)/x² at x = 0 using the
    polynomial-only variant.  The full taylor_remainder uses pt.switch
    over a closed branch (cos(Kx)-1)/x² whose k-th derivative is
    O(1/x^(2+k)) at x=0; switch is eager, so even though the polynomial
    branch is selected, the closed branch's NaN propagates.  The poly
    variant has no closed branch and is the right shape for iterated
    grads at the expansion point.
    """
    x = pt.dscalar("x")
    f = pt.cos(K * x)
    # Need order >= 6 to have x^5 term for the deepest grad; order=8 is
    # one extra term of slack with a much smaller graph than order=14.
    cur = taylor_remainder_poly(f, x, 0.0, 2, order=8)

    tol = 1e-10
    for k in range(6):
        out = float(pytensor.function([x], cur)(0.0))
        ref = _ref_grad_at_0_cos_Kx(K, k)
        ref_safe = abs(ref) if abs(ref) > 1e-300 else 1.0
        rel = abs(out - ref) / ref_safe
        assert rel <= tol, f"K={K} k={k}: got {out}, ref {ref}, rel_err {rel:.2e}"
        if k < 5:
            cur = pt.grad(cur, x)


# ---- switch-boundary continuity --------------------------------------------


@pytest.mark.parametrize("K", [1.0, 10.0, 100.0])
def test_n2_switch_boundary_continuity_cos_Kx(K):
    """Both branches must agree to ~tol at the eps switch boundary.
    A discontinuity here would mean either auto_eps is wrong or one of
    the branches is inaccurate near the boundary."""
    x = pt.dscalar("x")
    f = pt.cos(K * x)
    cache = TaylorAtPoint(f, x, 0.0)
    eps = auto_eps(cache, n=2, order=10)
    y = taylor_remainder(f, x, 0.0, 2, order=10, eps=eps)
    fn = pytensor.function([x], y)

    # Walk a tight neighborhood across the boundary.  At each point both
    # branches should evaluate to within 100·eps_machine relative.
    tol = 100 * np.finfo(np.float64).eps
    for frac in (0.999, 0.9999, 1.0001, 1.001):
        v = eps * frac
        ref = float(_R_cos_Kx(K, mp.mpf(v)))
        out = float(fn(v))
        rel = abs(out - ref) / max(1.0, abs(ref))
        assert rel <= tol, (
            f"K={K} v={v} eps={eps}: got {out}, ref {ref}, rel_err {rel:.2e}"
        )


# ---- pristine cos(K·x) - 1: same R as cos(K·x) but explicit f shape -------


@pytest.mark.parametrize("K", [1.0, 10.0, 100.0])
def test_n2_cosm1_pristine_form(K):
    """f = cos(K·x) - 1 has c_0 = 0, so P_1 = 0 and there is no K0
    cancellation. R = (cos(Kx) - 1)/x², same as for f = cos(K·x).
    Both forms should give identical answers up to ~eps_machine."""
    x = pt.dscalar("x")
    f_cos = pt.cos(K * x)
    f_cosm1 = pt.cos(K * x) - 1

    fn_cos = pytensor.function([x], taylor_remainder(f_cos, x, 0.0, 2, order=10))
    fn_cosm1 = pytensor.function([x], taylor_remainder(f_cosm1, x, 0.0, 2, order=10))

    tol = 100 * np.finfo(np.float64).eps
    for v in [0.0, 1e-12, 1e-6, 1e-3, 0.01, 0.1, 0.3]:
        out_cos = float(fn_cos(v))
        out_cosm1 = float(fn_cosm1(v))
        ref = float(_R_cos_Kx(K, mp.mpf(v)))
        ref_safe = abs(ref) if abs(ref) > 1e-300 else 1.0
        rel_cos = abs(out_cos - ref) / ref_safe
        rel_cosm1 = abs(out_cosm1 - ref) / ref_safe
        assert rel_cos <= tol and rel_cosm1 <= tol, (
            f"K={K} v={v}: cos-form rel_err {rel_cos:.2e}, cosm1-form rel_err {rel_cosm1:.2e}"
        )


# ---- auto_eps scale invariance under cos(K·x) ------------------------------


def test_n2_auto_eps_invariant_K_times_eps():
    """For f = cos(K·x), auto_eps should scale exactly as 1/K -- i.e.
    K · auto_eps is invariant. This is the natural eps in the variable
    y = K·x where the Taylor coefficients of cos(y) don't depend on K."""
    x = pt.dscalar("x")
    K_eps_pairs = []
    for K in [1e-3, 1.0, 10.0, 100.0, 1000.0]:
        f = pt.cos(K * x)
        cache = TaylorAtPoint(f, x, 0.0)
        eps = auto_eps(cache, n=2, order=10)
        K_eps_pairs.append((K, K * eps))

    # All K·eps values should agree to machine precision.
    refs = [v for _, v in K_eps_pairs]
    rel_spread = (max(refs) - min(refs)) / refs[0]
    assert rel_spread < 1e-12, f"K·eps not invariant: {K_eps_pairs}"
