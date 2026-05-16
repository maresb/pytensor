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


def test_stable_smooth_sinc_second_derivative_at_zero():
    """For sinc, sinc''(0) = -1/3.  With derivative_depth=2 the auto-chosen
    polynomial order = 3 leaves enough terms to survive two grads, and
    pt.grad through the switch picks up the polynomial branch's derivative
    at x=0."""
    from taylor_remainder import stable_smooth

    x = pt.dscalar("x")
    sinc = stable_smooth(pt.sin(x), x, 0.0, denominator_degree=1, derivative_depth=2)
    sinc_dd = pt.grad(pt.grad(sinc, x), x)
    fn = pytensor.function([x], sinc_dd)
    assert math.isclose(float(fn(0.0)), -1.0 / 3.0, abs_tol=1e-13)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
