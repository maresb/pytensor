"""Per-dtype calibration of auto_eps's `safety` and `tol_rel` parameters.

For each (dtype, function, n, order):
  - Find empirical eps* = largest |t| at which polynomial branch achieves
    relative accuracy <= 10·eps_machine (validated via mpmath reference).
  - Compute the unsafetied analytic prediction
        eps_analytic = (tol_rel · |c_n / c_{n+order}|)^(1/order)
    with `tol_rel = 10·eps_machine`.
  - Compute implied_safety = eps* / eps_analytic.

If implied_safety is consistent (across function/order) per dtype, then a
single safety constant per dtype is justified. We currently use safety=0.75
calibrated only on float64; this script verifies it for float16/32 too and
checks whether a tighter formulation (e.g. tol_rel = eps_machine + safety = 1)
would do as well.
"""

import math

import mpmath as mp
import numpy as np


mp.mp.dps = 50


def coeffs_log1p(K):
    return [mp.mpf(0)] + [(-1) ** (k + 1) * mp.mpf(1) / k for k in range(1, K)]


def coeffs_expm1(K):
    out = [mp.mpf(0)]
    inv = mp.mpf(1)
    for k in range(1, K):
        inv = inv / k
        out.append(inv)
    return out


def coeffs_sin(K):
    out = [mp.mpf(0)]
    inv = mp.mpf(1)
    for k in range(1, K):
        inv = inv / k
        if k % 2 == 1:
            m = (k - 1) // 2
            out.append(((-1) ** m) * inv)
        else:
            out.append(mp.mpf(0))
    return out


def coeffs_cosm1(K):
    out = [mp.mpf(0)]
    inv = mp.mpf(1)
    for k in range(1, K):
        inv = inv / k
        if k % 2 == 0 and k >= 2:
            m = k // 2
            out.append(((-1) ** m) * inv)
        else:
            out.append(mp.mpf(0))
    return out


CASES = [
    ("log1p_n1", coeffs_log1p(40), np.log1p, 1),
    ("expm1_n1", coeffs_expm1(40), np.expm1, 1),
    ("sin_n1", coeffs_sin(40), np.sin, 1),
    ("cosm1_n2", coeffs_cosm1(40), lambda x: np.cos(x) - 1, 2),
]


def first_nonzero(coeffs_full, start, max_count=5):
    for k in range(start, min(start + max_count, len(coeffs_full))):
        v = abs(float(coeffs_full[k]))
        if v != 0.0:
            return v, k
    return 0.0, start + max_count


def empirical_eps_below_tol(coeffs_full, np_f, n, order, dtype, tol_rel):
    """Largest |t| at which polynomial branch's rel-error stays <= tol_rel."""
    ts = np.logspace(-15, np.log10(0.5), 250)
    cs = np.array([float(c) for c in coeffs_full[n : n + order]], dtype=dtype)

    def poly(t):
        out = np.full_like(np.asarray(t, dtype=dtype), cs[-1])
        for c in cs[-2::-1]:
            out = out * np.asarray(t, dtype=dtype) + c
        return out

    # Compare poly to the "tail" Taylor series sum c_n + c_{n+1} t + ... at
    # high precision -- this is mathematically (f(t) - P_{n-1}(t)) / t^n.
    high_prec_tail = np.array(
        [
            float(
                sum(
                    coeffs_full[k] * mp.mpf(t) ** (k - n)
                    for k in range(n, len(coeffs_full))
                )
            )
            for t in ts
        ]
    )
    poly_vals = poly(ts).astype(np.float64)
    safe = np.where(np.abs(high_prec_tail) > 0, np.abs(high_prec_tail), 1.0)
    rel = np.abs(poly_vals - high_prec_tail) / safe
    rel[~np.isfinite(rel)] = 1.0

    below = np.where(rel <= tol_rel)[0]
    return ts[below[-1]] if len(below) else float("nan")


def analytic_eps_unsafetied(coeffs_full, n, order, tol_rel):
    """auto_eps formula with safety=1.0."""
    v_lead, k_lead = first_nonzero(coeffs_full, n)
    if v_lead == 0.0:
        return 1.0
    v_trunc, k_trunc = first_nonzero(coeffs_full, n + order)
    if v_trunc == 0.0:
        return 1.0
    return (tol_rel * v_lead / v_trunc) ** (1.0 / (k_trunc - k_lead))


def main():
    orders = [4, 6, 8, 10, 12, 14]
    dtypes = [np.float64, np.float32, np.float16]

    print(
        f"{'dtype':<8} {'function':<12} {'order':>5}  "
        f"{'tol_rel':>10}  {'eps_emp':>10} {'eps_analytic':>14}  "
        f"{'implied_safety':>15}"
    )
    print("-" * 90)
    rows = []
    for dtype in dtypes:
        eps_machine = float(np.finfo(dtype).eps)
        tol_rel = 10.0 * eps_machine
        for ident, coeffs_mp, np_f, n in CASES:
            for order in orders:
                if n + order >= len(coeffs_mp):
                    continue
                eps_emp = empirical_eps_below_tol(
                    coeffs_mp, np_f, n, order, dtype, tol_rel
                )
                eps_an = analytic_eps_unsafetied(coeffs_mp, n, order, tol_rel)
                if math.isnan(eps_emp) or eps_an == 0:
                    implied = float("nan")
                else:
                    implied = eps_emp / eps_an
                rows.append(
                    (dtype.__name__, ident, order, tol_rel, eps_emp, eps_an, implied)
                )
                print(
                    f"{dtype.__name__:<8} {ident:<12} {order:>5}  "
                    f"{tol_rel:>10.2e}  {eps_emp:>10.4g} {eps_an:>14.4g}  "
                    f"{implied:>15.3g}"
                )

    # Aggregate implied_safety per dtype
    print("\n" + "-" * 90)
    print(
        f"{'dtype':<8}  {'median safety':>14}  {'min safety':>12} {'max safety':>12}  "
        f"{'count':>6}"
    )
    print("-" * 90)
    for dtype in dtypes:
        vals = [r[6] for r in rows if r[0] == dtype.__name__ and not math.isnan(r[6])]
        if vals:
            v = sorted(vals)
            median = v[len(v) // 2]
            print(
                f"{dtype.__name__:<8}  {median:>14.4g}  {min(v):>12.4g} {max(v):>12.4g}  "
                f"{len(v):>6}"
            )

    print("\nNote: current code uses `safety=0.75` for all dtypes.")


if __name__ == "__main__":
    main()
