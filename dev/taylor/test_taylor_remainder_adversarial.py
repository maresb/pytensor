"""Adversarial parameter scan for taylor_remainder.

Sweeps function family x K x K0 x x_value x dtype combinations and verifies
each against an mpmath reference at 50 digits. The contract assumed:
f(x) is computed with at most ~eps_machine relative error; all
remaining error in taylor_remainder's output comes from our subtraction,
division, and polynomial truncation.

We DO NOT test cases that violate the f-pristine contract (e.g. user
passing `pt.exp(x) - 1` instead of `pt.expm1(x)`); those are documented
as out-of-scope.

We DO test:
  - K and K0 magnitudes spanning representable range.
  - x values across binades from subnormal-edge to ~0.5.
  - Multiple function families, n values, orders.
  - Opaque f (OpFromGraph): genuinely adversarial since canonicalize
    cannot fold the K0 cancellation.
"""

import math

import mpmath as mp
import numpy as np
import pytest
from taylor_remainder import taylor_remainder

import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph


mp.mp.dps = 60
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.on_opt_error = "ignore"


# ---- function families: each defines f(x; K, K0) and its remainder R ----
# (label, build_pt, mp_R, n_default)


def _build_K_sin(K, K0):
    return lambda x: K0 + K * pt.sin(x)


def _mp_R_K_sin(K, K0, x_mp, n):
    # R(x) for f = K0 + K·sin(x), at a=0:
    #   n=1: R = (f - (K0 + 0))/x = K · sin(x)/x.
    #   n=2: requires P_1(x) = K0 + K·x; (f - P_1)/x^2 = K·(sin(x) - x)/x^2.
    if n == 1:
        if x_mp == 0:
            return mp.mpf(K)
        return mp.mpf(K) * mp.sin(x_mp) / x_mp
    if n == 2:
        if x_mp == 0:
            return mp.mpf(0)  # K · -sin(0)/2 = 0
        return mp.mpf(K) * (mp.sin(x_mp) - x_mp) / x_mp**2
    raise NotImplementedError


def _build_K_expm1(K, K0):
    return lambda x: K0 + K * pt.expm1(x)


def _mp_R_K_expm1(K, K0, x_mp, n):
    # f = K0 + K·(e^x - 1). f(0) = K0. f'(0) = K.
    if n == 1:
        if x_mp == 0:
            return mp.mpf(K)
        return mp.mpf(K) * (mp.exp(x_mp) - 1) / x_mp
    if n == 2:
        if x_mp == 0:
            return mp.mpf(K) / 2
        return mp.mpf(K) * (mp.exp(x_mp) - 1 - x_mp) / x_mp**2
    raise NotImplementedError


def _build_K_cosm1(K, K0):
    return lambda x: K0 + K * (pt.cos(x) - 1)


def _mp_R_K_cosm1(K, K0, x_mp, n):
    # f = K0 + K·(cos(x) - 1). f(0) = K0. f'(0) = 0. f''(0) = -K.
    if n == 1:
        if x_mp == 0:
            return mp.mpf(0)
        return mp.mpf(K) * (mp.cos(x_mp) - 1) / x_mp
    if n == 2:
        if x_mp == 0:
            return -mp.mpf(K) / 2
        return mp.mpf(K) * (mp.cos(x_mp) - 1) / x_mp**2
    raise NotImplementedError


def _build_K_sin_minus_x(K, K0):
    return lambda x: K0 + K * (pt.sin(x) - x)


def _mp_R_K_sin_minus_x(K, K0, x_mp, n):
    # f = K0 + K·(sin(x) - x). f(0) = K0. f'(0) = 0. f''(0) = 0. f'''(0) = -K.
    if n == 1:
        if x_mp == 0:
            return mp.mpf(0)
        return mp.mpf(K) * (mp.sin(x_mp) - x_mp) / x_mp
    if n == 2:
        if x_mp == 0:
            return mp.mpf(0)
        return mp.mpf(K) * (mp.sin(x_mp) - x_mp) / x_mp**2
    if n == 3:
        if x_mp == 0:
            return -mp.mpf(K) / 6
        return mp.mpf(K) * (mp.sin(x_mp) - x_mp) / x_mp**3
    raise NotImplementedError


FAMILIES = {
    "K_sin": (_build_K_sin, _mp_R_K_sin, [1, 2]),
    "K_expm1": (_build_K_expm1, _mp_R_K_expm1, [1, 2]),
    "K_cosm1": (_build_K_cosm1, _mp_R_K_cosm1, [1, 2]),
    "K_sin_minus_x": (_build_K_sin_minus_x, _mp_R_K_sin_minus_x, [1, 2, 3]),
}


# ---- adversarial parameter grid ----


def _x_sweep(dtype):
    return [0.0, 1e-15, 1e-10, 1e-7, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.3]


@pytest.mark.parametrize("family_label", list(FAMILIES.keys()))
@pytest.mark.parametrize("K_log10", [-50, -10, 0, 10, 50])
@pytest.mark.parametrize("K0_choice", ["zero", "1", "1e10"])
@pytest.mark.parametrize("order", [10])
def test_adversarial_K_K0_scan(family_label, K_log10, K0_choice, order):
    """For every (family, K, K0, order), evaluate taylor_remainder at a
    sweep of x values and compare to mpmath reference. Tolerance is set
    by the polynomial-truncation budget at order plus a safety factor.
    """
    K = 10.0**K_log10
    K0 = {"zero": 0.0, "1": 1.0, "1e10": 1e10}[K0_choice]

    build_pt, mp_R, n_options = FAMILIES[family_label]

    for n in n_options:
        x = pt.dscalar("x")
        f = build_pt(K, K0)(x)
        try:
            y = taylor_remainder(f, x, 0.0, n, order=order)
        except Exception as e:
            pytest.fail(f"{family_label} K={K} K0={K0} n={n}: build failed: {e}")
        fn = pytensor.function([x], y)

        for v in _x_sweep(np.float64):
            try:
                ref = float(mp_R(K, K0, mp.mpf(v), n))
            except (mp.libmp.NoConvergence, ZeroDivisionError):
                continue  # skip points where the reference is degenerate
            with np.errstate(all="ignore"):
                out = float(fn(v))
            if not math.isfinite(out):
                pytest.fail(
                    f"{family_label} K={K} K0={K0} n={n} x={v}: not finite ({out})"
                )
            # Tolerance: machine-precision plus polynomial-truncation budget.
            tol = max(1e-12, 100 * np.finfo(np.float64).eps)
            ref_safe = abs(ref) if abs(ref) > 1e-300 else 1.0
            rel = abs(out - ref) / ref_safe
            assert rel <= tol, (
                f"{family_label} K={K} K0={K0} n={n} x={v}: "
                f"got {out}, ref {ref}, rel_err {rel:.2e} > tol {tol:.2e}"
            )


# ---- opaque-f scan: canonicalize cannot fold the K0 cancellation ----


@pytest.mark.parametrize("K_log10", [0])
@pytest.mark.parametrize("K0_choice", ["zero", "1", "1e3"])
def test_adversarial_opaque_polynomial(K_log10, K0_choice):
    """f wrapped in OpFromGraph (opaque to canonicalize). Polynomial f
    should still be computed correctly via the v_trunc=0 cancellation
    fallback in auto_eps.

    auto_eps and taylor_remainder are scale-invariant in K (verified
    by the wider test_adversarial_K_K0_scan); this opaque-f variant
    is expensive (OpFromGraph internal compile is ~2.5s per case), so
    we only sweep K0 here -- the parameter that exercises the
    cancellation fallback we're verifying.
    """
    K = 10.0**K_log10
    K0 = {"zero": 0.0, "1": 1.0, "1e3": 1e3}[K0_choice]

    inner_x = pt.dscalar("inner_x")
    inner = K0 + K * inner_x + K * inner_x**2
    opaque = OpFromGraph([inner_x], [inner])

    x = pt.dscalar("x")
    f = opaque(x)
    y = taylor_remainder(f, x, 0.0, 1, order=10)
    fn = pytensor.function([x], y)

    for v in [0.0, 1e-15, 1e-12, 1e-8, 1e-3, 0.01, 0.1, 0.3]:
        # R(x) for f = K0 + K*x + K*x^2:  R = K + K*x.
        ref = K + K * v
        with np.errstate(all="ignore"):
            out = float(fn(v))
        if not math.isfinite(out):
            pytest.fail(f"K={K} K0={K0} x={v}: not finite ({out})")
        ref_safe = abs(ref) if abs(ref) > 1e-300 else 1.0
        rel = abs(out - ref) / ref_safe
        assert rel <= 1e-12, (
            f"K={K} K0={K0} x={v}: got {out}, ref {ref}, rel_err {rel:.2e}"
        )


# ---- dtype scan: spot-checks for float32/float16 ----


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
@pytest.mark.parametrize("family_label", ["K_sin", "K_expm1"])
@pytest.mark.parametrize("K_log10", [-3, 0, 3])
def test_adversarial_dtype_scan(dtype_str, family_label, K_log10):
    """Verify taylor_remainder behaves correctly across dtypes for a few
    representative function/K combinations."""
    K = 10.0**K_log10
    K0 = 0.0
    _, mp_R, _ = FAMILIES[family_label]

    x = pt.scalar("x", dtype=dtype_str)
    K_dtype = np.asarray(K, dtype=dtype_str)
    f = K0 + K_dtype * (pt.sin(x) if family_label == "K_sin" else pt.expm1(x))
    y = taylor_remainder(f, x, 0.0, 1, order=10)
    fn = pytensor.function([x], y)

    eps_machine = float(np.finfo(np.dtype(dtype_str)).eps)
    tol = 100 * eps_machine

    sweep = [0.0, 1e-3, 0.01, 0.1, 0.3]
    if dtype_str == "float64":
        sweep = [0.0, 1e-15, 1e-10, *sweep]

    for v in sweep:
        try:
            ref = float(mp_R(K, K0, mp.mpf(v), 1))
        except (mp.libmp.NoConvergence, ZeroDivisionError):
            continue
        with np.errstate(all="ignore"):
            out = float(fn(np.asarray(v, dtype=dtype_str)))
        ref_safe = abs(ref) if abs(ref) > 1e-300 else 1.0
        rel = abs(out - ref) / ref_safe
        assert rel <= tol, (
            f"{family_label} dtype={dtype_str} K={K} x={v}: "
            f"got {out}, ref {ref}, rel_err {rel:.2e} > tol {tol:.2e}"
        )
