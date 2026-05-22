"""Test suite for taylor_remainder. Run with `pytest test_taylor_remainder.py`.

Reference values are produced by mpmath at 50 decimal digits.
"""

import math
import warnings

import mpmath as mp
import numpy as np
import pytest
from taylor_remainder import (
    TaylorAtPoint,
    TaylorRemainderOverflowWarning,
    TaylorRemainderUnderflowWarning,
    auto_eps,
    closed_branch_needed,
    taylor_remainder,
    taylor_remainder_poly,
)

import pytensor
import pytensor.tensor as pt
from pytensor.graph.rewriting.utils import rewrite_graph


mp.mp.dps = 50
pytensor.config.mode = "FAST_COMPILE"


# ------------ reference --------------------------------------------------


def mp_remainder(mp_f, n, t):
    """High-precision n-th Taylor remainder of f at 0 evaluated at t."""
    t_mp = mp.mpf(float(t))
    if t_mp == 0:
        # R(0) = c_n
        return float(mp.taylor(mp_f, 0, n)[n])
    f_t = mp_f(t_mp)
    if n == 0:
        return float(f_t)
    P = mp.mpf(0)
    pow_t = mp.mpf(1)
    for c in mp.taylor(mp_f, 0, n - 1):
        P = P + c * pow_t
        pow_t = pow_t * t_mp
    return float((f_t - P) / t_mp**n)


# (id, build_f_pt(x), mp_f, n)
CASES = [
    ("log1p", lambda x: pt.log1p(x), mp.log1p, 1),
    ("expm1", lambda x: pt.expm1(x), lambda t: mp.exp(t) - 1, 1),
    ("sin", lambda x: pt.sin(x), mp.sin, 1),
    ("cosm1_n2", lambda x: pt.cos(x) - 1, lambda t: mp.cos(t) - 1, 2),
    ("sin_minus_x_n3", lambda x: pt.sin(x) - x, lambda t: mp.sin(t) - t, 3),
]


# ------------ forward evaluation ----------------------------------------


@pytest.mark.parametrize("ident,f_pt,f_mp,n", CASES, ids=[c[0] for c in CASES])
def test_taylor_remainder_matches_mpmath(ident, f_pt, f_mp, n):
    x = pt.dscalar("x")
    y = taylor_remainder(f_pt(x), x, 0.0, n, order=10)
    fn = pytensor.function([x], y)
    for v in [-0.5, -1e-8, 0.0, 1e-8, 0.5]:
        ref = mp_remainder(f_mp, n, v)
        out = float(fn(v))
        assert math.isclose(out, ref, rel_tol=1e-12, abs_tol=1e-15), (
            f"{ident}@x={v}: got {out}, expected {ref}"
        )


def test_x_equals_a_returns_finite_limit_value():
    """The switch must select the polynomial branch at x=a, not produce NaN."""
    x = pt.dscalar("x")
    y = taylor_remainder(pt.log1p(x), x, 0.0, 1)
    out = float(pytensor.function([x], y)(0.0))
    assert math.isfinite(out) and math.isclose(out, 1.0, abs_tol=1e-15)


# ------------ iterated gradient -----------------------------------------


def test_iterated_grad_log1p_psi1():
    """d^k/dx^k [log1p(x)/x] at x=0  =  (-1)^k k!/(k+1).

    Use the polynomial-only variant: iterated grad of a polynomial is just
    a shorter polynomial, so this tests derivative correctness without
    the closed-branch quotient-rule blow-up.
    """
    x = pt.dscalar("x")
    cur = taylor_remainder_poly(pt.log1p(x), x, 0.0, 1, order=12)
    cur = rewrite_graph(cur, include=("canonicalize",))
    for k in range(8):
        ref = (-1) ** k * math.factorial(k) / (k + 1)
        out = float(pytensor.function([x], cur)(0.0))
        assert math.isclose(out, ref, rel_tol=1e-10), f"k={k}"
        cur = pt.grad(cur, x)
        cur = rewrite_graph(cur, include=("canonicalize",))


# ------------ scale invariance -------------------------------------------


@pytest.mark.parametrize("K", [1e-50, 1e-3, 1.0, 1e3, 1e50])
def test_auto_eps_scale_invariant(K):
    """auto_eps(K·f) = auto_eps(f) for any nonzero constant K."""
    x = pt.dscalar("x")
    eps_unit = auto_eps(TaylorAtPoint(pt.sin(x), x, 0.0), n=1, order=10)
    eps_scaled = auto_eps(TaylorAtPoint(K * pt.sin(x), x, 0.0), n=1, order=10)
    assert math.isclose(eps_unit, eps_scaled, rel_tol=1e-12)


@pytest.mark.parametrize("K", [1e-50, 1e50])
def test_taylor_remainder_evaluates_correctly_under_extreme_scaling(K):
    x = pt.dscalar("x")
    y = taylor_remainder(K * pt.sin(x), x, 0.0, 1, order=10)
    fn = pytensor.function([x], y)
    for v in [0.0, 1e-8, 0.1, 0.5]:
        ref = K * mp_remainder(mp.sin, 1, v)
        out = float(fn(v))
        # Scaled by K, but relative precision should be machine-level.
        assert math.isclose(out, ref, rel_tol=1e-12, abs_tol=K * 1e-300)


# ------------ dtype awareness --------------------------------------------


def test_auto_eps_widens_for_lower_precision_dtype():
    x32 = pt.scalar("x", dtype="float32")
    x64 = pt.scalar("x", dtype="float64")
    eps32 = auto_eps(TaylorAtPoint(pt.sin(x32), x32, 0.0), n=1, order=10)
    eps64 = auto_eps(TaylorAtPoint(pt.sin(x64), x64, 0.0), n=1, order=10)
    assert eps32 > eps64
    # eps grows like eps_machine^(1/order); eps_machine ratio ~ 5e8 -> eps ratio ~ 7
    assert 3 < eps32 / eps64 < 15


# ------------ closed_branch_needed ---------------------------------------


def test_closed_branch_needed_for_log1p_within_unit_circle():
    x = pt.dscalar("x")
    cache = TaylorAtPoint(pt.log1p(x), x, 0.0)
    # log1p has slow series convergence; closed branch is needed even at order 14.
    assert closed_branch_needed(cache, n=1, order=14, t_max=0.5)


def test_closed_branch_unneeded_for_well_converging_series():
    x = pt.dscalar("x")
    # sinc has factorial-decay coefficients; at order=12 polynomial alone covers
    # |t| <= 0.5 to better than machine precision.
    cache = TaylorAtPoint(pt.sin(x), x, 0.0)
    assert not closed_branch_needed(cache, n=1, order=14, t_max=0.5)


# ------------ vanishing-coefficient corner cases -------------------------


def _f64_from_bits(biased_exp, mantissa):
    import struct

    bits = ((biased_exp & 0x7FF) << 52) | (mantissa & ((1 << 52) - 1))
    return struct.unpack("!d", struct.pack("!Q", bits))[0]


def _f32_from_bits(biased_exp, mantissa):
    import struct

    bits = ((biased_exp & 0xFF) << 23) | (mantissa & ((1 << 23) - 1))
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def test_naive_sinc_correct_for_every_nonzero_float():
    """For both float32 and float64, sweep every biased exponent crossed with
    four mantissa patterns (zero, all-ones, alt 0101, alt 1010). The only
    representable x for which naive sin(x)/x fails to match mpmath is x=0
    exactly -- verified across ~9k structurally interesting floats.
    """
    f64_patterns = [0x0, 0xFFFFFFFFFFFFF, 0x5555555555555, 0xAAAAAAAAAAAAA]
    f32_patterns = [0x0, 0x7FFFFF, 0x555555, 0x2AAAAA]

    for dtype, from_bits, exp_max, patterns in [
        (np.float64, _f64_from_bits, 0x7FF - 1, f64_patterns),
        (np.float32, _f32_from_bits, 0xFF - 1, f32_patterns),
    ]:
        eps_machine = float(np.finfo(dtype).eps)
        for biased_exp in range(0, exp_max + 1):
            for mantissa in patterns:
                x_py = from_bits(biased_exp, mantissa)
                if x_py == 0 or not math.isfinite(x_py):
                    continue
                with np.errstate(all="ignore"):
                    naive = float(
                        np.sin(np.asarray(x_py, dtype=dtype))
                        / np.asarray(x_py, dtype=dtype)
                    )
                ref = float(mp.sin(mp.mpf(x_py)) / mp.mpf(x_py))
                assert math.isfinite(naive), (
                    f"{dtype.__name__} bias={biased_exp} m={mantissa:x}: not finite"
                )
                rel = abs(naive - ref) / max(1.0, abs(ref))
                assert rel <= 100 * eps_machine, (
                    f"{dtype.__name__} bias={biased_exp} m={mantissa:x} x={x_py:.3e}: rel_err={rel:.2e}"
                )


@pytest.mark.parametrize("dtype_str", ["float16", "float32", "float64"])
def test_underflow_warning_for_extreme_subnormal_c_n(dtype_str):
    """When the leading coefficient is small enough that the closed branch
    can underflow within the polynomial-branch window, taylor_remainder
    should warn -- regardless of dtype.

    K = 12·smallest_subnormal places c_1 just above the subnormal floor and
    c_3 = -K/6 = -2·smallest_subnormal also just above; the formula at
    order=2 then gives an eps for which |c_1|·eps < 10·smallest_subnormal,
    triggering the warning. We use order=2 so c_{n+order}=c_3 doesn't itself
    underflow during coefficient evaluation.
    """
    ss = float(np.finfo(np.dtype(dtype_str)).smallest_subnormal)
    K = 12 * ss
    x = pt.scalar("x", dtype=dtype_str)
    f = np.asarray(K, dtype=dtype_str) * pt.sin(x)
    with pytest.warns(TaylorRemainderUnderflowWarning):
        taylor_remainder(f, x, 0.0, 1, order=2)


def test_no_underflow_warning_for_normal_c_n():
    """For typical c_n ≥ 1e-300 or so, no warning should be raised."""
    x = pt.dscalar("x")
    for K in [1.0, 1e-50, 1e-100, 1e-200, 1e-300, 1e50]:
        f = K * pt.sin(x)
        with warnings.catch_warnings():
            warnings.simplefilter("error", TaylorRemainderUnderflowWarning)
            taylor_remainder(f, x, 0.0, 1, order=10)  # would raise if warned


def test_no_underflow_warning_when_c_n_vanishes():
    """If f vanishes to higher order than n (so c_n = 0), no warning --
    the situation is "this expression has nothing of order n at x=a"."""
    x = pt.dscalar("x")
    # f = x^3, n = 1: c_1 = 0 exactly. Skip the check.
    f = x**3
    with warnings.catch_warnings():
        warnings.simplefilter("error", TaylorRemainderUnderflowWarning)
        taylor_remainder(f, x, 0.0, 1, order=10)


def test_overflow_warning_for_large_c_n_with_eps_above_one():
    """When eps > 1 and |c_n| is near largest_finite, the polynomial-branch
    leading term overflows. (Force eps explicitly since auto_eps picks
    something < 1 for f64.)"""
    x = pt.dscalar("x")
    K = 1e308  # near largest_finite for float64
    f = K * pt.sin(x)
    # eps=1.5 is above 1; combined with c_n=K ~ max_float, c_n·eps overflows
    with pytest.warns(TaylorRemainderOverflowWarning):
        taylor_remainder(f, x, 0.0, 1, order=10, eps=1.5)


def test_no_overflow_warning_for_normal_c_n():
    """For typical c_n with auto_eps, no overflow warning."""
    x = pt.dscalar("x")
    for K in [1.0, 1e-50, 1e50, 1e150, 1e300]:
        f = K * pt.sin(x)
        with warnings.catch_warnings():
            warnings.simplefilter("error", TaylorRemainderOverflowWarning)
            taylor_remainder(f, x, 0.0, 1, order=10)


def test_n1_polynomial_f_with_nonzero_constant_term():
    """f(x) = K0 + K*x + L*x^2 with K0 != 0, n=1.

    R(x) = (f - K0)/x = K + L*x.  c_0=K0, c_1=K, c_2=L; c_3 onward vanish so
    auto_eps lands on the eps=0 path. The closed expression  (f - K0)/x
    looks numerically dangerous (catastrophic cancellation when |K0|
    dominates), but pytensor's canonicalize folds the K0 - K0 symbolically
    before evaluation -- so closed reduces to K + L*x and stays accurate at
    arbitrarily small x.

    Verified across several K0 magnitudes including 1e10 and 1e-10.
    """
    x = pt.dscalar("x")
    K, L = 2.0, 3.0
    for K0 in [0.0, 1.0, 1e10, 1e-10, -1e5]:
        f = K0 + K * x + L * x**2
        y = taylor_remainder(f, x, 0.0, 1, order=10)
        fn = pytensor.function([x], y)
        for v in [0.0, 1e-15, 1e-10, 1e-3, 0.5]:
            ref = K + L * v
            out = float(fn(v))
            assert math.isclose(out, ref, rel_tol=1e-12, abs_tol=1e-15), (
                f"K0={K0}, x={v}: got {out}, expected {ref}"
            )


def test_n1_transcendental_f_with_nonzero_constant_term():
    """f(x) = K0 + exp(x) at a=0 with K0 != 0, n=1.

    R(x) = ((K0 + exp(x)) - (K0 + 1))/x = (exp(x) - 1)/x = expm1(x)/x.

    Hits the formula path (v_trunc = c_11 = 1/11! != 0). Polynomial
    branch covers |x| < eps_upper ≈ 0.148 (independent of K0); for
    larger |x| we use the closed form, where pytensor's canonicalize
    folds the (K0 - K0) cancellation symbolically -- so even huge K0
    doesn't introduce numerical cancellation.
    """
    x = pt.dscalar("x")
    for K0 in [0.0, 1.0, 1e10, -1e5]:
        f = K0 + pt.exp(x)
        y = taylor_remainder(f, x, 0.0, 1, order=10)
        fn = pytensor.function([x], y)
        for v in [0.0, 1e-15, 1e-10, 1e-3, 0.5]:
            ref = 1.0 if v == 0 else math.expm1(v) / v
            out = float(fn(v))
            assert math.isclose(out, ref, rel_tol=1e-12, abs_tol=1e-15), (
                f"K0={K0}, x={v}: got {out}, expected {ref}"
            )


def test_n1_cancellation_aware_eps_with_opaque_f_and_nonzero_c0():
    """When f is opaque to canonicalize (wrapped in OpFromGraph), the
    closed form (f - K0)/x is NOT folded: it computes f(x) and subtracts
    K0 at runtime, with catastrophic cancellation at small |x| when
    K0 != 0. For this case the polynomial-truncation formula degenerates
    (v_trunc = 0), so auto_eps must fall back to the cancellation-aware
    bound

        eps_cancel  =  (eps_machine · |c_0| / (tol_rel · |c_{k_lead}|))^(1/k_lead)

    For the OpFromGraph-wrapped 1 + 2x + 3x^2 case at n=1: c_0=1, c_1=2,
    k_lead=1, tol_rel = 2·(order+1)·eps_machine = 22·eps_machine for
    order=10.  So eps_cancel = 1/(22·2) = 1/44.

    Regression: earlier code returned eps=0 in this case, sending all
    small-x evaluations through the cancellation-prone closed form
    (rel_err 8e-4 at x=1e-15).
    """
    from taylor_remainder import _tol_rel

    from pytensor.compile.builders import OpFromGraph

    inner_x = pt.dscalar("inner_x")
    inner_expr = 1.0 + 2.0 * inner_x + 3.0 * inner_x**2
    opaque_f = OpFromGraph([inner_x], [inner_expr])

    x = pt.dscalar("x")
    f = opaque_f(x)

    # Use the explicit-coefficients constructor mode: bypasses auto_eps's
    # slow grad chain through OpFromGraph (~1.5s for order=10) since the
    # polynomial's coefficients are known.
    cache = TaylorAtPoint(f, x, 0.0, coefficients=[1.0, 2.0, 3.0] + [0.0] * 13)
    eps = auto_eps(cache, n=1, order=10)
    eps_machine = float(np.finfo(np.float64).eps)
    tol_rel = _tol_rel(10, np.float64)
    expected = (eps_machine * 1.0 / (tol_rel * 2.0)) ** (1.0 / 1)
    assert math.isclose(eps, expected, rel_tol=1e-12), (
        f"expected eps={expected}, got {eps}"
    )

    y = taylor_remainder(f, x, 0.0, 1, order=10, cache=cache)
    fn = pytensor.function([x], y)
    for v in [0.0, 1e-15, 1e-10, 1e-5, 1e-3, 0.5, 1.0]:
        ref = 2.0 + 3.0 * v
        out = float(fn(v))
        assert math.isclose(out, ref, rel_tol=1e-12, abs_tol=1e-15), (
            f"x={v}: got {out}, expected {ref}"
        )


def test_iterated_grad_at_a_in_eps_zero_path():
    """For f = x + x^3, n=1: R(x) = (x + x^3)/x = 1 + x^2. With order=10
    the higher coefficients vanish, so auto_eps -> 0. The eps=0 switch must
    still expose the polynomial's higher-order derivatives at x=a, not just
    the leading constant.

    Regression: an earlier `pt.switch(t==0, c_n_const, closed)` returned
    R''(0) = 0 because the constant has zero gradients. Fixed by switching
    to the full polynomial expression so d^k/dx^k recovers k! * c_{n+k}.
    """
    x = pt.dscalar("x")
    f = x + x**3
    cur = taylor_remainder(f, x, 0.0, 1, order=10)
    assert auto_eps(TaylorAtPoint(f, x, 0.0), n=1, order=10) == 0.0
    expected = {0: 1.0, 1: 0.0, 2: 2.0, 3: 0.0}
    for k, ref in expected.items():
        out = float(pytensor.function([x], cur)(0.0))
        assert math.isclose(out, ref, abs_tol=1e-12), (
            f"k={k}: got {out}, expected {ref}"
        )
        cur = pt.grad(cur, x)


def test_polynomial_window_with_all_higher_orders_vanishing():
    """If f is itself a polynomial with no nonzero higher-order coeffs,
    auto_eps returns 0.0 -- meaning "defer to user's f for x != a"."""
    x = pt.dscalar("x")
    f = x**3  # all derivatives beyond order 3 are zero
    cache = TaylorAtPoint(f, x, 0.0)
    eps = auto_eps(cache, n=3, order=10)
    assert eps == 0.0


def test_auto_eps_fallback_uses_k_lead_for_sparse_leading_coefficient():
    """auto_eps's v_trunc=0 fallback uses 1/k_lead (not 1/n) so that
    sparse-leading-coefficient cases get the correct exponent.

    For f = K0 + K2·x² (a polynomial, so v_trunc = 0 for any order >= 1)
    at n=1: c_0 = K0, c_1 = 0, c_2 = K2.  So k_lead = 2, not 1.
    The cancellation-aware bound is
        eps_lower = (eps_machine · |c_0| / (tol_rel · |c_2|))^(1/2)
                  = sqrt(eps_machine · K0 / (tol_rel · K2)).

    The earlier bug used `1/n = 1/1`, returning eps_machine·K0/(tol_rel·K2)
    -- a much smaller eps that hurt nothing for polynomial f (canonicalize
    folds the closed branch) but would mis-size the polynomial window for
    transcendental f with the same sparse-leading shape.
    """
    from taylor_remainder import _tol_rel

    K0, K2 = 1.0, 3.0
    x = pt.dscalar("x")
    f = K0 + K2 * x**2  # c_1 = 0, k_lead = 2
    cache = TaylorAtPoint(f, x, 0.0)
    eps = auto_eps(cache, n=1, order=10)

    eps_machine = float(np.finfo(np.float64).eps)
    tol_rel = _tol_rel(10, np.float64)
    expected = (eps_machine * K0 / (tol_rel * K2)) ** (1.0 / 2)
    assert math.isclose(eps, expected, rel_tol=1e-12), (
        f"expected eps={expected} from 1/k_lead formula, got {eps}"
    )


def test_explicit_coefficients_finite_list_raises_on_overrun():
    """A finite coefficient list should raise IndexError -- not silently
    extrapolate -- when the cache asks for an index past the list end.
    Counterpart: an infinite generator (e.g. itertools.repeat(0.0)) is
    fine because next() never raises StopIteration."""
    x = pt.dscalar("x")
    f = pt.cos(x)  # symbolic f doesn't matter -- explicit mode ignores it for coeffs

    cache = TaylorAtPoint(f, x, 0.0, coefficients=[1.0, 0.0, -0.5])  # only c_0..c_2

    # Indices 0..2 are fine.
    assert cache.numeric_value_at_a(0) == 1.0
    assert cache.numeric_value_at_a(2) == math.factorial(2) * -0.5

    # Index 3 should raise -- list exhausted.
    with pytest.raises(IndexError, match="exhausted"):
        cache.numeric_value_at_a(3)


def test_explicit_coefficients_deriv_returns_polynomial_truncation():
    """In explicit mode, deriv(m) returns the m-th derivative of the
    polynomial truncation Σ c_k (x-a)^k -- a symbolic pytensor expression
    that is exact at x=a and accurate near x=a."""
    x = pt.dscalar("x")
    # User's symbolic f is irrelevant for the cache's deriv in explicit mode.
    f = pt.cos(x)
    # Coefficients of, say, sin(x): c_1=1, c_3=-1/6, c_5=1/120, ...
    coeffs = [0.0, 1.0, 0.0, -1.0 / 6, 0.0, 1.0 / 120]
    cache = TaylorAtPoint(f, x, 0.0, coefficients=coeffs)

    def at_zero(expr):
        # deriv(m) for m near the polynomial degree may collapse to a
        # constant after canonicalize, leaving x as an "unused input".
        return float(pytensor.function([x], expr, on_unused_input="ignore")(0.0))

    # deriv(0) at x=0 should give c_0 = 0
    assert math.isclose(at_zero(cache.deriv(0)), 0.0)
    # deriv(1) at x=0 should give 1!*c_1 = 1
    assert math.isclose(at_zero(cache.deriv(1)), 1.0)
    # deriv(3) at x=0 should give 3!*c_3 = -1
    assert math.isclose(at_zero(cache.deriv(3)), -1.0)
    # deriv(5) at x=0 should give 5!*c_5 = 1
    assert math.isclose(at_zero(cache.deriv(5)), 1.0)

    # And deriv(6) -- past the supplied coefficients -- raises
    with pytest.raises(IndexError, match="exhausted"):
        cache.deriv(6)


def test_min_order_and_eps_finds_minimum_sufficient_order():
    """`_min_order_and_eps` grows `order` lazily from `derivative_depth + 1`
    until both poly truncation and closed-branch cancellation meet
    `tol_rel`. For pristine, well-conditioned f the minimum is exactly
    `derivative_depth + 1`. For cancelled f the order grows enough to
    widen eps until the |v|^{-c} amplification fits within tol_rel."""
    from taylor_remainder import _min_order_and_eps

    x = pt.dscalar("x")

    # sinc = R_1(sin), pristine: order tracks depth + 1 exactly.
    cache = TaylorAtPoint(pt.sin(x), x, 0.0)
    for depth in range(6):
        order, eps = _min_order_and_eps(cache, n=1, derivative_depth=depth)
        assert order == depth + 1, f"depth={depth}: got order={order}"
        assert eps >= 0.0

    # Same R_2 numerator with two different declared cancellation_orders --
    # c=0 (wrong, just to check the parameter is honored) gives a small
    # order; c=2 forces a much wider polynomial window.
    cache = TaylorAtPoint(x * pt.cos(x) - pt.sin(x), x, 0.0)
    order_c0, eps_c0 = _min_order_and_eps(cache, n=2, cancellation_order=0)
    order_c2, eps_c2 = _min_order_and_eps(cache, n=2, cancellation_order=2)
    assert order_c2 > order_c0, (
        f"c=2 should grow order beyond c=0 case: got c0={order_c0}, c2={order_c2}"
    )
    assert eps_c2 > eps_c0, f"c=2 should yield wider eps: got c0={eps_c0}, c2={eps_c2}"


def test_closed_branch_rel_err_bound_amplifies_with_cancellation_order():
    """`cancellation_order=c` says the user's f evaluates with
    rel_err ≤ ε_m·|v|^{-c}.  The closed-branch bound should amplify
    the floor by |v|^{-c} for small v (and be unchanged for c=0)."""
    from taylor_remainder import closed_branch_rel_err_bound

    # Coefficients of sin(x): the P_{n=1}-subtraction sum is 0 (c_0 = 0),
    # so for c=0 the bound is exactly the ε_m floor. For c>0 the floor
    # amplifies by |v|^{-c}.
    coeffs = [
        (-1) ** ((k - 1) // 2) / math.factorial(k) if k % 2 == 1 else 0.0
        for k in range(20)
    ]
    x = pt.dscalar("x")
    cache = TaylorAtPoint(x, x, 0.0, coefficients=coeffs)

    eps_machine = float(np.finfo(np.float64).eps)
    for v in (0.5, 0.1, 0.01):
        b0 = closed_branch_rel_err_bound(cache, n=1, v=v, order=8, cancellation_order=0)
        b1 = closed_branch_rel_err_bound(cache, n=1, v=v, order=8, cancellation_order=1)
        b2 = closed_branch_rel_err_bound(cache, n=1, v=v, order=8, cancellation_order=2)
        # c=0 baseline: just the ε_m floor (sin's c_0 = 0 so no P-subtraction).
        assert math.isclose(b0, eps_machine, rel_tol=1e-12)
        # c>0 amplification: each unit of c multiplies the bound by 1/v.
        assert math.isclose(b1, eps_machine / v, rel_tol=1e-12)
        assert math.isclose(b2, eps_machine / v**2, rel_tol=1e-12)


def test_stable_smooth_sinc_forward():
    """stable_smooth(sin, x, 0, denominator_degree=1) = sinc.  Forward eval
    matches at x=0 (limit value 1) and away from zero (sin(x)/x)."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    sinc = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
    fn = pytensor.function([x], sinc)
    assert math.isclose(float(fn(0.0)), 1.0, abs_tol=1e-15)
    assert math.isclose(float(fn(0.5)), math.sin(0.5) / 0.5, rel_tol=1e-14)
    assert math.isclose(float(fn(1.0)), math.sin(1.0) / 1.0, rel_tol=1e-14)


def test_stable_smooth_sinc_iterated_grad_at_zero():
    """sinc^(k)(0) for k = 0, 1, 2, 3, 4.  Reference values come from the
    closed form  sinc^(k)(0) = (-1)^(k/2) / (k+1) for even k, 0 for odd k.
    The grad chain via pullback constructs another stable_smooth at each
    grad; correctness across the chain is the point here.  Pure perf at
    higher depths is covered separately by
    test_stable_smooth_depth5_under_wallclock_budget."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    cur = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
    expected = [1.0, 0.0, -1.0 / 3.0, 0.0, 1.0 / 5.0]
    for k, ref in enumerate(expected):
        fn = pytensor.function([x], cur)
        v = float(fn(0.0))
        assert math.isclose(v, ref, abs_tol=1e-12), f"k={k}: got {v}, expected {ref}"
        cur = pt.grad(cur, x)


def test_stable_smooth_n3_sin_minus_x_forward():
    """(sin(x) - x) / x^3 = -1/6 + x^2/120 - ...  At x=0 limit is -1/6.
    Exercises denominator_degree=3 (3-term subtraction in P_{n-1})."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    f = stable_smooth(pt.sin(x) - x, x, 0.0, denominator_degree=3)
    fn = pytensor.function([x], f)
    assert math.isclose(float(fn(0.0)), -1.0 / 6.0, abs_tol=1e-15)
    for t in (0.1, 0.5, 1.0):
        expected = (math.sin(t) - t) / t**3
        assert math.isclose(float(fn(t)), expected, rel_tol=1e-12)


def test_stable_smooth_n3_sin_minus_x_first_grad():
    """d/dx [(sin(x) - x) / x^3] at x=0: by Taylor, the series is
    -1/6 + x^2/120 - x^4/5040 + ..., so derivative at 0 is 0.
    Exercises the n=3 pullback path, which recurses with stable_smooth
    of denominator_degree=2."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    f = stable_smooth(pt.sin(x) - x, x, 0.0, denominator_degree=3)
    fp = pt.grad(f, x)
    fn = pytensor.function([x], fp)
    assert math.isclose(float(fn(0.0)), 0.0, abs_tol=1e-13)
    # And at moderate x: mpmath ref.
    for t in (0.1, 0.5, 1.0):
        ref = float(mp.diff(lambda u: (mp.sin(u) - u) / u**3, mp.mpf(t), 1))
        assert math.isclose(float(fn(t)), ref, rel_tol=1e-10)


def test_stable_smooth_n2_cosm1_forward():
    """(cos(x) - 1)/x^2 = -1/2 + x^2/24 - x^4/720 + ...; at x=0 the limit
    is -1/2.  Uses denominator_degree=2 with a pristine numerator (c=0)."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    f = stable_smooth(pt.cos(x) - 1, x, 0.0, denominator_degree=2)
    fn = pytensor.function([x], f)
    assert math.isclose(float(fn(0.0)), -0.5, abs_tol=1e-15)
    for t in (0.1, 0.5, 1.0):
        expected = (math.cos(t) - 1) / t**2
        assert math.isclose(float(fn(t)), expected, rel_tol=1e-12)


def test_stable_smooth_n2_cosm1_grad_at_zero():
    """d/dx [(cos(x)-1)/x^2] at x=0.  By series, (cos(x)-1)/x^2 = -1/2
    + x^2/24 - ..., so derivative at 0 is 0.  Exercises the n>1 pullback
    (R_{n-1}(f') = R_1(-sin(x)) path)."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    f = stable_smooth(pt.cos(x) - 1, x, 0.0, denominator_degree=2)
    fp = pt.grad(f, x)
    fn = pytensor.function([x], fp)
    assert math.isclose(float(fn(0.0)), 0.0, abs_tol=1e-14)


def test_stable_smooth_float32_propagates_dtype():
    """stable_smooth's auto-chosen eps must scale with x.dtype's eps_machine,
    so a float32 input gives a wider polynomial window than float64."""
    from taylor_remainder import stable_smooth

    x32 = pt.scalar("x", dtype="float32")
    sinc32 = stable_smooth(pt.sin(x32), x32, 0.0, denominator_degree=1)
    fn = pytensor.function([x32], sinc32)
    # Forward at small but nonzero x: polynomial branch handles it.
    v = float(fn(np.float32(1e-6)))
    # sin(1e-6)/1e-6 ~ 1 - (1e-6)^2/6 ~ 1 to float32 precision.
    assert math.isclose(v, 1.0, abs_tol=1e-6)
    # And at moderate x.
    t = np.float32(0.5)
    v = float(fn(t))
    assert math.isclose(v, float(math.sin(0.5) / 0.5), rel_tol=1e-6)


def test_stable_smooth_expansion_point_nonzero():
    """stable_smooth at a != 0: expand (sin(x) - sin(a))/(x - a) around a.
    This is f(x) := sin(x), n=1 expansion around `a`, giving a stable
    forward eval of (sin(x) - sin(a))/(x - a)."""
    from taylor_remainder import stable_smooth

    a = 1.7
    x = pt.dscalar("x")
    # Numerator must vanish at x = a (i.e., have c_0 = 0 in the local
    # expansion) for the n=1 remainder to be finite there.
    g = stable_smooth(pt.sin(x) - math.sin(a), x, a, denominator_degree=1)
    fn = pytensor.function([x], g)
    # At x = a, the value is the derivative: cos(a).
    assert math.isclose(float(fn(a)), math.cos(a), abs_tol=1e-14)
    # And at x close to a, matches (sin(x) - sin(a))/(x - a).
    for dx in (1e-7, 0.01, 0.5):
        t = a + dx
        expected = (math.sin(t) - math.sin(a)) / (t - a)
        assert math.isclose(float(fn(t)), expected, rel_tol=1e-10), (
            f"x={t}: got {float(fn(t))}, expected {expected}"
        )


def test_stable_smooth_sinc_second_grad_at_nonzero_points():
    """Confirm the grad chain stays accurate not just at x=0 (where the
    polynomial branch handles everything) but also at moderate x where
    the closed branch is in play through both levels of the chain.

    Reference: sinc''(t) = -2/t^3 * sin(t) + (2*cos(t))/t^2 - sin(t)/t.
    Compared to mpmath at 50 dps.
    """
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    sinc = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
    sinc_dd = pt.grad(pt.grad(sinc, x), x)
    fn = pytensor.function([x], sinc_dd)
    for t in (1e-6, 0.01, 0.1, 0.5, 1.0, 2.0):
        # mpmath's numerical 2nd derivative of sinc at t.
        expected = float(mp.diff(lambda u: mp.sin(u) / u, mp.mpf(t), 2))
        got = float(fn(t))
        assert math.isclose(got, expected, rel_tol=1e-10, abs_tol=1e-14), (
            f"sinc''({t}): got {got}, expected {expected}"
        )


def test_stable_smooth_grad_chain_at_nonzero_expansion_point():
    """Grad chain at a=1.7: d/dx (sin(x)-sin(a))/(x-a) at x=a should
    equal cos(a)'/2 = -sin(a)/2 (the second-derivative limit of the
    Newton-quotient at a)."""
    from taylor_remainder import stable_smooth

    a = 1.7
    x = pt.dscalar("x")
    f = stable_smooth(pt.sin(x) - math.sin(a), x, a, denominator_degree=1)
    fp = pt.grad(f, x)
    fn = pytensor.function([x], fp)
    # At x=a, the value of d/dx[(sin(x)-sin(a))/(x-a)] = sin''(a)/2 = -sin(a)/2
    # (limit of the Newton quotient's derivative).
    expected_at_a = -math.sin(a) / 2.0
    assert math.isclose(float(fn(a)), expected_at_a, abs_tol=1e-13)
    # At x close to a, matches mpmath ref.
    for dx in (1e-6, 0.05, 0.5):
        t = a + dx
        ref = float(
            mp.diff(
                lambda u: (mp.sin(u) - mp.sin(mp.mpf(a))) / (u - mp.mpf(a)),
                mp.mpf(t),
                1,
            )
        )
        assert math.isclose(float(fn(t)), ref, rel_tol=1e-10, abs_tol=1e-14), (
            f"grad at x={t}: got {float(fn(t))}, expected {ref}"
        )


def test_stable_smooth_grad_chain_in_float32():
    """float32: forward + first grad should give correct values within
    float32 precision."""
    from taylor_remainder import stable_smooth

    x32 = pt.scalar("x", dtype="float32")
    sinc32 = stable_smooth(pt.sin(x32), x32, 0.0, denominator_degree=1)
    grad32 = pt.grad(sinc32, x32)
    fn = pytensor.function([x32], grad32)
    # sinc'(0) = 0, sinc'(0.5) = (0.5*cos(0.5) - sin(0.5))/0.25.
    assert math.isclose(float(fn(np.float32(0.0))), 0.0, abs_tol=1e-6)
    expected = (0.5 * math.cos(0.5) - math.sin(0.5)) / 0.25
    assert math.isclose(float(fn(np.float32(0.5))), expected, rel_tol=1e-5)


def test_stable_smooth_grad_through_cancellation_order_2_numerator():
    """When a cancelled numerator (cancellation_order > 0) gets
    differentiated, the chain's bracket inherits c+1, which compounds
    through subsequent grads.  Smoke test: pt.grad of
    stable_smooth(x*cos-sin, x, 0, n=2, c=2) (which equals sinc'(x))
    should give sinc''(x) accurately."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    sinc_prime_cancelled = stable_smooth(
        x * pt.cos(x) - pt.sin(x),
        x,
        0.0,
        denominator_degree=2,
        cancellation_order=2,
    )
    sinc_dprime = pt.grad(sinc_prime_cancelled, x)
    fn = pytensor.function([x], sinc_dprime)
    for t in (0.0, 1e-6, 0.1, 0.5, 1.0):
        if t == 0.0:
            expected = -1.0 / 3.0
        else:
            expected = float(mp.diff(lambda u: mp.sin(u) / u, mp.mpf(t), 2))
        got = float(fn(t))
        assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-13), (
            f"sinc''({t}) via cancelled-numerator grad chain: got {got}, expected {expected}"
        )


def test_stable_smooth_grad_correct_in_underflow_neighborhood():
    """Design pitfall (1): `pt.switch(x==0, 1, sin(x)/x)` evaluates fine
    almost everywhere, but pt.grad chains through the `sin(x)/x` branch,
    whose quotient-rule derivative `(x*cos(x) - sin(x))/x^2` cancels in
    a float64 neighborhood of zero.  Whether pytensor's canonicalize
    rewrites the naive grad into a stable form is version-dependent, so
    we test stable_smooth's correctness directly across the
    cancellation-prone region rather than comparing to naive."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    safe = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
    fn = pytensor.function([x], pt.grad(safe, x))
    # mpmath ref across the polynomial-branch window for sinc.
    for t in (1e-12, 1e-9, 1e-7, 1e-4, 0.1):
        t_mp = mp.mpf(t)
        expected = float((t_mp * mp.cos(t_mp) - mp.sin(t_mp)) / t_mp**2)
        v = float(fn(t))
        assert math.isclose(v, expected, rel_tol=1e-10, abs_tol=1e-15), (
            f"t={t}: stable_smooth grad got {v}, expected {expected}"
        )


def test_stable_smooth_solves_constant_branch_zero_grad_pitfall():
    """Design pitfall (2): `pt.switch(x==0, 1.0, expm1(x)/x)` has the
    correct forward value (the literal switch returns 1 at x=0 and
    expm1(x)/x elsewhere, both equal to the true limit).  But pt.grad
    at x=0 propagates through the *constant* branch, returning 0 -- the
    correct answer is 1/2.

    stable_smooth's polynomial branch retains the correct local
    Taylor structure, so its gradient at x=0 is 1/2 as expected."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    naive = pt.switch(pt.eq(x, 0), 1.0, pt.expm1(x) / x)
    safe = stable_smooth(pt.expm1(x), x, 0.0, denominator_degree=1)

    fn_naive_grad = pytensor.function([x], pt.grad(naive, x))
    fn_safe_grad = pytensor.function([x], pt.grad(safe, x))

    # d/dx [expm1(x)/x] at x=0 = 1/2  (from expm1(x)/x = 1 + x/2 + x^2/6 + ...).
    v_naive = float(fn_naive_grad(0.0))
    v_safe = float(fn_safe_grad(0.0))

    # naive: constant branch's grad is 0 -- wrong.
    assert v_naive == 0.0, (
        f"expected naive grad at 0 to be 0 (the pitfall), got {v_naive}"
    )
    # safe: correct value 1/2.
    assert math.isclose(v_safe, 0.5, abs_tol=1e-13), (
        f"stable_smooth grad at 0: got {v_safe}, expected 0.5"
    )


def test_stable_smooth_f_over_g_composition():
    """Design composition pattern: when f(a) = g(a) = 0 to the same order
    k, the ratio f(x)/g(x) is stable at x=a via

        stable_smooth(f, x, a, denominator_degree=k) /
        stable_smooth(g, x, a, denominator_degree=k).

    Test case: sin(x)/tan(x) = cos(x). Both vanish at 0 to order 1.
    """
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    R_sin = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
    R_tan = stable_smooth(pt.tan(x), x, 0.0, denominator_degree=1)
    ratio = R_sin / R_tan
    fn = pytensor.function([x], ratio)
    for t in (0.0, 1e-9, 0.01, 0.5, 1.0):
        expected = math.cos(t)
        assert math.isclose(float(fn(t)), expected, rel_tol=1e-12, abs_tol=1e-14), (
            f"sin/tan@t={t}: got {float(fn(t))}, expected {expected}"
        )


def test_stable_smooth_n2_cancellation_order_2_matches_sinc_prime():
    """The user encodes sinc'(x) as `(x cos x - sin x) / x^2` with
    `cancellation_order=2` (the literal subtraction loses 2 orders of
    precision in the numerator at small x).  Result should match
    `pt.grad(sinc(x), x)` -- both compute sinc'(x) -- and the closed
    form `(t cos t - sin t)/t^2` away from zero."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    cancelled = stable_smooth(
        x * pt.cos(x) - pt.sin(x),
        x,
        0.0,
        denominator_degree=2,
        cancellation_order=2,
    )
    sinc_prime = pt.grad(stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1), x)
    fn_a = pytensor.function([x], cancelled)
    fn_b = pytensor.function([x], sinc_prime)
    # NB: don't compute the reference via float math.sin/math.cos --
    # `(t*cos(t) - sin(t))/t^2` is exactly the cancellation `stable_smooth`
    # is supposed to *avoid*, so float-level reference would compare two
    # things both right (in different ways).  Use mpmath at 50 dps.
    for t in (0.0, 1e-8, 0.01, 0.1, 0.5, 1.0):
        t_mp = mp.mpf(float(t))
        expected = (
            0.0 if t == 0 else float((t_mp * mp.cos(t_mp) - mp.sin(t_mp)) / t_mp**2)
        )
        v_a = float(fn_a(t))
        v_b = float(fn_b(t))
        assert math.isclose(v_a, expected, abs_tol=1e-15, rel_tol=1e-10), (
            f"cancelled@t={t}: got {v_a}, expected {expected}"
        )
        assert math.isclose(v_b, expected, abs_tol=1e-15, rel_tol=1e-12), (
            f"grad(sinc)@t={t}: got {v_b}, expected {expected}"
        )


def test_stable_smooth_inline_paths_agree_numerically():
    """inline=True (the default) and inline=False compile via different
    paths -- inlined-at-build-time vs lazy-per-OFG -- and the perf
    profile is very different, but the numerical output must be
    identical."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    sinc_inline = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1, inline=True)
    sinc_lazy = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1, inline=False)
    fn_i = pytensor.function([x], sinc_inline)
    fn_l = pytensor.function([x], sinc_lazy)
    for t in (0.0, 0.1, 0.5, 1.0):
        v_i, v_l = float(fn_i(t)), float(fn_l(t))
        assert math.isclose(v_i, v_l, rel_tol=1e-14, abs_tol=1e-15), (
            f"t={t}: inline={v_i}, lazy={v_l}"
        )
    # Same with one grad: inline propagates through pullback's recursive call.
    fn_gi = pytensor.function([x], pt.grad(sinc_inline, x))
    fn_gl = pytensor.function([x], pt.grad(sinc_lazy, x))
    for t in (0.0, 0.1, 0.5):
        v_i, v_l = float(fn_gi(t)), float(fn_gl(t))
        assert math.isclose(v_i, v_l, rel_tol=1e-14, abs_tol=1e-15)


def test_stable_smooth_depth5_under_wallclock_budget():
    """Regression test for the depth-5 first-eval cliff (Task A in the
    follow-ups doc).  With the default inline=True and the upstream
    OFG-cloning fixes in place, build + function() + first eval +
    second eval for sinc grad-chain depth 5 should stay well under
    30 wall-clock seconds on any reasonable CI host.  Pre-fix this
    same configuration was >2 minutes.

    The budget is intentionally generous (3x typical local runtime,
    ~4x CI runtime) so the test catches a clear regression rather
    than flapping on minor perf jitter."""
    import time

    from taylor_remainder import stable_smooth

    t0 = time.perf_counter()
    x = pt.dscalar("x")
    cur = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
    for _ in range(5):
        cur = pt.grad(cur, x)
    fn = pytensor.function([x], cur)
    fn(0.0)
    fn(0.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, (
        f"depth-5 first-eval budget exceeded: {elapsed:.1f}s > 30.0s. "
        f"This is the cliff Task A in dev/taylor/stable_smooth_followups.md "
        f"was meant to prevent regressing -- check whether OFG cloning "
        f"behavior in pytensor.compile.builders changed."
    )


def test_stable_smooth_n0_is_passthrough():
    """denominator_degree=0 returns the numerator directly (no
    OpFromGraph wrap). R_0(f) = f, so there's nothing to stabilize."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    f = pt.sin(x)
    result = stable_smooth(f, x, 0.0, denominator_degree=0)
    # Pass-through: same object back.
    assert result is f


def test_stable_smooth_negative_n_raises():
    """Negative denominator_degree is invalid."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    with pytest.raises(ValueError, match="non-negative"):
        stable_smooth(pt.sin(x), x, 0.0, denominator_degree=-1)


def test_stable_smooth_vector_input_sinc_forward():
    """Sinc evaluated entrywise on a vector input -- the entry-at-zero
    test in particular goes through the polynomial branch where naive
    sin(x)/x would NaN.  Tolerance is tight (rel_tol=1e-12) since the
    polynomial branch is essentially exact for sinc."""
    from taylor_remainder import stable_smooth

    x = pt.dvector("x")
    sinc = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
    fn = pytensor.function([x], sinc)
    out = fn(np.array([0.0, 1e-10, 1e-4, 0.5, 1.0, -0.3]))
    expected = np.array(
        [
            1.0,
            1.0,
            1.0 - 1e-8 / 6,
            math.sin(0.5) / 0.5,
            math.sin(1.0),
            math.sin(-0.3) / -0.3,
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)


def test_stable_smooth_vector_input_sinc_grad_matches_mpmath():
    """Elementwise gradient: pt.grad(sinc.sum(), x) recovers sinc'
    entrywise.  Uses mpmath at 50 dps to ground-truth the cancellation-
    prone near-zero entries."""
    from taylor_remainder import stable_smooth

    x = pt.dvector("x")
    sinc = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
    sinc_p = pt.grad(sinc.sum(), x)
    fn = pytensor.function([x], sinc_p)
    ts = np.array([0.0, 1e-12, 1e-8, 1e-4, 0.5, 1.0, -0.3])
    out = fn(ts)

    def sinc_prime(t):
        if t == 0:
            return 0.0
        return float((mp.mpf(t) * mp.cos(t) - mp.sin(t)) / mp.mpf(t) ** 2)

    expected = np.array([sinc_prime(t) for t in ts])
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)


def test_stable_smooth_vector_input_n2_cosm1():
    """(cos x - 1)/x^2 entrywise on a vector input, including x=0 where
    the limit is -1/2."""
    from taylor_remainder import stable_smooth

    x = pt.dvector("x")
    f = stable_smooth(pt.cos(x) - 1, x, 0.0, denominator_degree=2)
    fn = pytensor.function([x], f)
    ts = np.array([0.0, 1e-8, 1e-4, 0.5, 1.0])
    out = fn(ts)
    expected = np.array(
        [
            -0.5,
            (math.cos(1e-8) - 1) / 1e-16 if False else -0.5,
            (math.cos(1e-4) - 1) / 1e-8,
            (math.cos(0.5) - 1) / 0.25,
            (math.cos(1.0) - 1) / 1.0,
        ]
    )
    # Near-zero entries are reference-limited by float cancellation in
    # the expected formula too; use mpmath for those.
    expected[1] = float((mp.cos(mp.mpf(1e-8)) - 1) / mp.mpf(1e-8) ** 2)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)


def test_stable_smooth_vector_input_nonzero_expansion_point():
    """a != 0 with vector x: stable_smooth(sin(x) - sin(1.7), x, 1.7,
    denominator_degree=1) reproduces cos(1.7) at t = a and the closed
    form (sin t - sin a)/(t - a) elsewhere.  Tolerance and reference
    pattern mirrors the scalar test_stable_smooth_expansion_point_nonzero
    -- including using `math.sin` (not mpmath) for the reference, since
    the closed branch evaluates the same lossy subtraction and matches
    that to ~1e-10."""
    from taylor_remainder import stable_smooth

    a = 1.7
    x = pt.dvector("x")
    f = stable_smooth(pt.sin(x) - math.sin(a), x, a, denominator_degree=1)
    fn = pytensor.function([x], f)
    ts = np.array([a, a + 1e-7, a + 0.01, a + 0.5])
    out = fn(ts)

    assert math.isclose(out[0], math.cos(a), abs_tol=1e-14)
    for got, t in zip(out[1:], ts[1:]):
        expected = (math.sin(t) - math.sin(a)) / (t - a)
        assert math.isclose(got, expected, rel_tol=1e-10), (
            f"t={t}: got {got}, expected {expected}"
        )


def test_stable_smooth_vector_input_float32():
    """Vector + float32 dtype: the scalar surrogate carries x.dtype, and
    the auto-eps formula sizes itself to float32 epsilon."""
    from taylor_remainder import stable_smooth

    x = pt.fvector("x")
    sinc = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
    fn = pytensor.function([x], sinc)
    out = fn(np.array([0.0, 1e-4, 0.5, 1.0], dtype="float32"))
    assert out.dtype == np.float32
    expected = np.array(
        [1.0, math.sin(1e-4) / 1e-4, math.sin(0.5) / 0.5, math.sin(1.0)],
        dtype="float32",
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-7)


def test_stable_smooth_vector_input_grad_chain_depth2():
    """Second derivative through the vector grad path.  d^2 sinc/dx^2
    at x = 0 equals -1/3; closed form sinc''(t) elsewhere via mpmath."""
    from taylor_remainder import stable_smooth

    x = pt.dvector("x")
    sinc = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
    sinc_p = pt.grad(sinc.sum(), x)
    sinc_pp = pt.grad(sinc_p.sum(), x)
    fn = pytensor.function([x], sinc_pp)
    ts = np.array([0.0, 1e-6, 0.3, 1.0])
    out = fn(ts)

    def sinc_pp_ref(t):
        if t == 0:
            return -1.0 / 3.0
        t_mp = mp.mpf(t)
        sin_t, cos_t = mp.sin(t_mp), mp.cos(t_mp)
        # sinc''(t) = ((2 - t^2) sin t - 2 t cos t) / t^3
        return float(((2 - t_mp**2) * sin_t - 2 * t_mp * cos_t) / t_mp**3)

    expected = np.array([sinc_pp_ref(t) for t in ts])
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-14)


def test_stable_smooth_vector_input_cache_raises():
    """Cross-call cache sharing isn't implemented for vector inputs (the
    cache's scalar surrogate would need a different keying scheme).
    Reject loudly rather than silently miscompute."""
    from taylor_remainder import TaylorAtPoint, stable_smooth

    x = pt.dscalar("x")
    cache = TaylorAtPoint(pt.sin(x), x, 0.0)
    x_v = pt.dvector("x")
    with pytest.raises(NotImplementedError, match="cache= currently requires scalar x"):
        stable_smooth(pt.sin(x_v), x_v, 0.0, denominator_degree=1, cache=cache)


def test_numeric_value_at_a_rejects_free_symbolic_inputs():
    """Auto-eps and _min_order_and_eps need a Python float per Taylor
    coefficient; if the numerator depends on a free symbolic input besides
    x, that float doesn't exist (`.eval()` has no value to plug in).
    Reject loudly rather than letting pt.grad fail downstream with a
    less obvious message."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    z = pt.dscalar("z")
    # `log1p(x*z)`'s first derivative at x=0 is z, a free leaf -- not a
    # number we can bake into eps/order.
    with pytest.raises(ValueError, match="non-constant leaf"):
        stable_smooth(pt.log1p(x * z), x, 0.0, denominator_degree=1)


def test_numeric_value_at_a_rejects_shared_inputs():
    """A shared variable in the numerator is the more insidious case: at
    construction time it has *some* value, so `.eval()` would silently
    return a float and we'd bake eps/order from it.  Later mutations to
    the shared input would update the symbolic polynomial branch but
    not the thresholds, which is a soundness bug.  Reject at
    construction."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    z = pytensor.shared(np.float64(2.0))
    with pytest.raises(ValueError, match="non-constant leaf"):
        stable_smooth(pt.log1p(x * z), x, 0.0, denominator_degree=1)


def test_stable_smooth_vector_non_elementwise_numerator_raises():
    """The pullback recovers the per-entry derivative via
    `pt.grad(num.sum(), x)`, which is correct only when the Jacobian is
    diagonal -- i.e., the numerator is elementwise in x.  Hand it a
    numerator that mixes entries (pt.sum, pt.dot, fancy indexing, ...)
    and the gradient is silently wrong; the structural check rejects
    these at construction so the user sees a clear error rather than
    bad numbers."""
    from taylor_remainder import stable_smooth

    x = pt.dvector("x")
    # pt.sum(pt.sin(x)) is a scalar; ones_like(x) broadcasts it to vector.
    # Each output entry depends on every input entry -- non-diagonal Jacobian.
    non_elem = pt.sum(pt.sin(x)) * pt.ones_like(x)
    with pytest.raises(NotImplementedError, match="not elementwise"):
        stable_smooth(non_elem, x, 0.0, denominator_degree=1)


def test_stable_smooth_vector_ones_like_allowed():
    """`pt.ones_like(x)` uses Alloc + Shape internally -- non-Elemwise --
    but is morally elementwise (each entry independent of x's values, just
    its shape).  The whitelist allows shape-only ops, so a numerator like
    `pt.sin(x) + pt.ones_like(x)` should construct cleanly."""
    from taylor_remainder import stable_smooth

    x = pt.dvector("x")
    # Adding ones_like(x) shifts every entry by 1; sin(x) is elementwise.
    # Net effect: elementwise (sin(x)+1)/x with limit 1+1=... wait
    # sin(x)/x -> 1 at 0, plus ones_like(x)/x -> infinite.  Skip the
    # divide here; this test is purely about the structural check passing.
    f = pt.sin(x) + pt.ones_like(x) - 1  # subtract back to keep c_0 = 0
    g = stable_smooth(f, x, 0.0, denominator_degree=1)
    fn = pytensor.function([x], g)
    out = fn(np.array([0.0, 0.5, 1.0]))
    # f(x)/x = (sin x + 1 - 1)/x = sinc(x); same numerics as the plain
    # sinc test (the ones_like contributes nothing after the -1).
    expected = np.array([1.0, math.sin(0.5) / 0.5, math.sin(1.0)])
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)


def test_stable_smooth_sinc_grad_at_nonzero_points():
    """Away from x=0, pt.grad of stable_smooth(sin, n=1) must still give
    correct sinc' values.  Closed-form: sinc'(t) = (t cos t - sin t) / t^2."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    sinc = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
    sinc_p = pt.grad(sinc, x)
    fn = pytensor.function([x], sinc_p)
    for t in (0.1, 0.5, 1.0, 2.0):
        expected = (t * math.cos(t) - math.sin(t)) / t**2
        got = float(fn(t))
        assert math.isclose(got, expected, rel_tol=1e-12), (
            f"t={t}: got {got}, expected {expected}"
        )


# ------------ stable_smooth shared cache (Task C) ------------------------


def test_stable_smooth_shared_cache_matches_independent_results():
    """Numeric forward output is identical whether stable_smooth builds its
    own TaylorAtPoint cache or the user supplies one."""
    from taylor_remainder import TaylorAtPoint, stable_smooth

    x = pt.dscalar("x")
    sin_x = pt.sin(x)
    cache = TaylorAtPoint(sin_x, x, 0.0)

    sinc_shared = stable_smooth(sin_x, x, 0.0, denominator_degree=1, cache=cache)
    sinc_indep = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1)
    fn_shared = pytensor.function([x], sinc_shared)
    fn_indep = pytensor.function([x], sinc_indep)
    for t in (0.0, 1e-12, 1e-8, 0.5, 1.7, -0.3):
        assert float(fn_shared(t)) == float(fn_indep(t))

    # And the cache works across different denominator_degrees too.
    sin_minus_x = pt.sin(x) - x
    cache2 = TaylorAtPoint(sin_minus_x, x, 0.0)
    r3 = stable_smooth(sin_minus_x, x, 0.0, denominator_degree=3, cache=cache2)
    fn3 = pytensor.function([x], r3)
    # (sin t - t)/t^3 = -1/6 + t^2/120 - ...; at t=0 the limit is -1/6.
    assert math.isclose(float(fn3(0.0)), -1.0 / 6.0, abs_tol=1e-15)


def test_stable_smooth_shared_cache_reduces_grad_calls():
    """The whole point of the shared-cache path: when two stable_smooth
    calls share a cache, the pt.grad-driven coefficient chain runs once
    instead of twice. Verify by counting calls to taylor_remainder.pt.grad
    (which TaylorAtPoint.deriv goes through) during construction."""
    from taylor_remainder import TaylorAtPoint, stable_smooth
    import taylor_remainder as _tr

    counter = [0]
    orig_grad = _tr.pt.grad

    def counting_grad(*args, **kwargs):
        counter[0] += 1
        return orig_grad(*args, **kwargs)

    _tr.pt.grad = counting_grad
    try:
        x = pt.dscalar("x")
        sin_x = pt.sin(x)
        counter[0] = 0
        stable_smooth(sin_x, x, 0.0, denominator_degree=1)
        stable_smooth(sin_x, x, 0.0, denominator_degree=2)
        independent_count = counter[0]

        x = pt.dscalar("x")
        sin_x = pt.sin(x)
        counter[0] = 0
        cache = TaylorAtPoint(sin_x, x, 0.0)
        stable_smooth(sin_x, x, 0.0, denominator_degree=1, cache=cache)
        stable_smooth(sin_x, x, 0.0, denominator_degree=2, cache=cache)
        shared_count = counter[0]
    finally:
        _tr.pt.grad = orig_grad

    assert shared_count < independent_count, (
        f"shared cache did not reduce pt.grad work: "
        f"shared={shared_count}, independent={independent_count}"
    )


def test_stable_smooth_cache_x_mismatch_raises():
    """Supplying a cache built over a different `x` variable is a silent
    correctness hazard, so we reject it loudly."""
    from taylor_remainder import TaylorAtPoint, stable_smooth

    x = pt.dscalar("x")
    y = pt.dscalar("y")
    cache = TaylorAtPoint(pt.sin(x), x, 0.0)
    with pytest.raises(ValueError, match="cache.x is not the same variable"):
        stable_smooth(pt.sin(x), y, 0.0, denominator_degree=1, cache=cache)


def test_stable_smooth_cache_a_mismatch_raises():
    """Supplying a cache built at a different expansion point is rejected."""
    from taylor_remainder import TaylorAtPoint, stable_smooth

    x = pt.dscalar("x")
    cache = TaylorAtPoint(pt.sin(x), x, 0.0)
    with pytest.raises(ValueError, match="cache.a"):
        stable_smooth(pt.sin(x), x, 1.0, denominator_degree=1, cache=cache)


def test_stable_smooth_cache_numerator_mismatch_raises():
    """Supplying a cache built over a structurally-different numerator is
    rejected. Two pt.sin(x) calls produce distinct objects but structurally
    match -- only a genuinely different graph should error."""
    from taylor_remainder import TaylorAtPoint, stable_smooth

    x = pt.dscalar("x")
    sin_cache = TaylorAtPoint(pt.sin(x), x, 0.0)
    # cos vs sin: real structural mismatch.
    with pytest.raises(ValueError, match="not structurally equal"):
        stable_smooth(pt.cos(x), x, 0.0, denominator_degree=1, cache=sin_cache)
    # Same expression rebuilt: equal_computations should accept it.
    stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1, cache=sin_cache)


# ------------ derived-expression composition (GEV link example) ------------


def _gev_link_pt(xi, z):
    """Pytensor expression for (1 + xi*z)^(-1/xi) via the design's
    derived-expression composition pattern.  Used by several tests below
    -- factored out so the worked example matches the docstring in
    `stable_smooth_design.md` line-for-line."""
    from pytensor.graph.replace import clone_replace
    from taylor_remainder import stable_smooth

    u = pt.scalar(dtype=xi.dtype) if xi.ndim == 0 else pt.vector(dtype=xi.dtype)
    h_of_u = stable_smooth(pt.log1p(u), u, 0.0, denominator_degree=1)
    h_of_xz = clone_replace(h_of_u, {u: xi * z})
    return pt.exp(-z * h_of_xz)


def _gev_link_mp(xi, z):
    """Reference value for (1 + xi*z)^(-1/xi) at 50 dps; takes the
    Gumbel limit at xi = 0."""
    xi_m, z_m = mp.mpf(xi), mp.mpf(z)
    if xi_m == 0:
        return float(mp.exp(-z_m))
    return float((1 + xi_m * z_m) ** (-1 / xi_m))


def test_gev_link_forward_across_xi_zero():
    """(1 + xi*z)^(-1/xi) is well-defined for all real xi on its domain.
    At xi=0 the value is exp(-z) (Gumbel).  Naive evaluation gives
    0/0-form NaN in the exponent or `1**inf` indeterminate; the
    composition pattern using stable_smooth(log1p/u) recovers the
    correct limit and remains accurate for tiny xi where the naive
    formula loses digits to cancellation."""
    xi = pt.dscalar("xi")
    z = pt.dscalar("z")
    f = _gev_link_pt(xi, z)
    fn = pytensor.function([xi, z], f)

    for xi_v, z_v in [
        (0.0, 0.5),
        (0.0, 1.0),
        (0.0, -0.3),
        (1e-15, 0.5),
        (1e-8, 0.5),
        (-1e-8, 0.5),
        (0.1, 0.5),
        (0.5, 0.5),
        (-0.3, 0.5),
        (1.0, 0.1),
        (-0.5, 0.5),
    ]:
        got = float(fn(xi_v, z_v))
        ref = _gev_link_mp(xi_v, z_v)
        assert math.isclose(got, ref, rel_tol=1e-12, abs_tol=1e-15), (
            f"xi={xi_v}, z={z_v}: got {got}, expected {ref}"
        )


def test_gev_link_d_dxi_correct_through_xi_zero():
    """pt.grad w.r.t. xi flows through the OFG pullback plus pytensor's
    outer chain rule (d(xi*z)/dxi = z).  Reference: by Taylor expansion
    of log f at xi=0,
        df/dxi |_{xi=0}  =  (z^2 / 2) * exp(-z).
    Away from zero, compare to mpmath central differences at h=1e-4."""
    xi = pt.dscalar("xi")
    z = pt.dscalar("z")
    f = _gev_link_pt(xi, z)
    df_dxi = pt.grad(f, xi)
    fn = pytensor.function([xi, z], df_dxi)

    for z_v in (0.3, 0.5, 1.0, -0.4):
        got = float(fn(0.0, z_v))
        expected = 0.5 * z_v**2 * math.exp(-z_v)
        assert math.isclose(got, expected, rel_tol=1e-12), (
            f"xi=0, z={z_v}: got {got}, expected {expected}"
        )

    def mp_d_dxi(xi_v, z_v):
        h = mp.mpf("1e-12")
        xi_m, z_m = mp.mpf(xi_v), mp.mpf(z_v)
        plus = (1 + (xi_m + h) * z_m) ** (-1 / (xi_m + h))
        minus = (1 + (xi_m - h) * z_m) ** (-1 / (xi_m - h))
        return float((plus - minus) / (2 * h))

    # Compare at moderate xi where mpmath central diff is stable.
    for xi_v, z_v in [(0.1, 0.5), (0.3, 0.5), (-0.3, 0.5), (0.5, 0.5)]:
        got = float(fn(xi_v, z_v))
        ref = mp_d_dxi(xi_v, z_v)
        assert math.isclose(got, ref, rel_tol=1e-8), (
            f"xi={xi_v}, z={z_v}: got {got}, expected {ref}"
        )


def test_gev_link_d_dz_correct_through_xi_zero():
    """pt.grad w.r.t. z: outer chain rule gives d(xi*z)/dz = xi, and the
    closed-form derivative is
        df/dz  =  -(1 + xi*z)^(-1/xi - 1).
    At xi=0 the limit is -exp(-z)."""
    xi = pt.dscalar("xi")
    z = pt.dscalar("z")
    f = _gev_link_pt(xi, z)
    df_dz = pt.grad(f, z)
    fn = pytensor.function([xi, z], df_dz)

    for z_v in (0.3, 0.5, 1.0):
        got = float(fn(0.0, z_v))
        expected = -math.exp(-z_v)
        assert math.isclose(got, expected, rel_tol=1e-12), (
            f"xi=0, z={z_v}: got {got}, expected {expected}"
        )

    for xi_v, z_v in [(0.1, 0.5), (-0.3, 0.5), (0.5, 0.5)]:
        got = float(fn(xi_v, z_v))
        ref = -float((1 + mp.mpf(xi_v) * mp.mpf(z_v)) ** (-1 / mp.mpf(xi_v) - 1))
        assert math.isclose(got, ref, rel_tol=1e-12), (
            f"xi={xi_v}, z={z_v}: got {got}, expected {ref}"
        )


def test_gev_link_vector_xi_forward_and_grad():
    """Vector xi via a vector surrogate u: the composition pattern works
    identically, with elementwise broadcast against scalar z."""
    xi = pt.dvector("xi")
    z = pt.dscalar("z")
    f = _gev_link_pt(xi, z)
    fn = pytensor.function([xi, z], f)
    xi_vals = np.array([0.0, 1e-12, 0.1, 0.5, -0.3])
    z_v = 0.5
    out = fn(xi_vals, z_v)
    expected = np.array([_gev_link_mp(float(v), z_v) for v in xi_vals])
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)

    # Gradient w.r.t. xi (elementwise).
    df_dxi = pt.grad(f.sum(), xi)
    fn_g = pytensor.function([xi, z], df_dxi)
    out_g = fn_g(xi_vals, z_v)
    # At xi=0 and xi=1e-12: limit is z^2/2 * exp(-z).
    limit = 0.5 * z_v**2 * math.exp(-z_v)
    assert math.isclose(out_g[0], limit, rel_tol=1e-12)
    assert math.isclose(out_g[1], limit, rel_tol=1e-9)
    # Away from zero, gradient is finite and non-NaN.
    assert np.all(np.isfinite(out_g))


def test_gev_link_zero_z_returns_one():
    """At z=0 the function is identically 1 regardless of xi (including
    xi=0, where the limit calculation is non-trivial)."""
    xi = pt.dscalar("xi")
    z = pt.dscalar("z")
    f = _gev_link_pt(xi, z)
    fn = pytensor.function([xi, z], f)
    for xi_v in (0.0, 1e-10, 0.1, -0.3, 0.5):
        assert math.isclose(float(fn(xi_v, 0.0)), 1.0, abs_tol=1e-15)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
