"""Empirical sweep of taylor_remainder branch behavior across the knobs:
function, dtype, expansion order, and |t|.

Reports for each (function, dtype, order):
  - empirical precision crossover (smallest |t| where closed-form beats poly)
  - largest |t| at which poly stays within `tol = 10 * eps_machine`
  - per-element runtime (numpy proxy) of poly branch vs closed-form branch
  - analytic auto_eps prediction (with safety=0.75) for comparison

Reference: mpmath at 50 digits.

Note on runtime: pt.switch evaluates BOTH branches per element, so total cost
is poly + closed regardless of |t|. The runtime numbers here are useful for:
  (a) understanding compile-time / graph-size cost as `order` grows
  (b) deciding whether to drop the closed branch entirely (if poly suffices)
  (c) comparing to ifelse-based variants (out of scope here)
"""

import math
import time

import mpmath as mp
import numpy as np


mp.mp.dps = 50


# -------- Taylor coefficients (mpmath) --------


def coeffs_log1p_mp(K):
    return [mp.mpf(0)] + [(-1) ** (k + 1) * mp.mpf(1) / k for k in range(1, K)]


def coeffs_expm1_mp(K):
    out = [mp.mpf(0)]
    inv = mp.mpf(1)
    for k in range(1, K):
        inv = inv / k
        out.append(inv)
    return out


def coeffs_cosm1_mp(K):
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


# -------- evaluators --------


def eval_poly_dtype(coeffs_full, t, n, order, dtype):
    cs = np.array([float(c) for c in coeffs_full[n : n + order]], dtype=dtype)
    t = np.asarray(t, dtype=dtype)
    out = np.full_like(t, cs[-1])
    for c in cs[-2::-1]:
        out = out * t + c
    return out


def eval_closed_dtype(np_f, coeffs_full, t, n, dtype):
    t = np.asarray(t, dtype=dtype)
    f_t = np_f(t)
    if n == 0:
        return f_t
    cs = np.array([float(c) for c in coeffs_full[:n]], dtype=dtype)
    P = np.full_like(t, cs[-1])
    for c in cs[-2::-1]:
        P = P * t + c
    return (f_t - P) / t**n


def reference_value(mp_f, coeffs_full_mp, t, n):
    t_mp = mp.mpf(float(t))
    if n == 0:
        return mp_f(t_mp)
    P = mp.mpf(0)
    pow_t = mp.mpf(1)
    for k in range(n):
        P = P + coeffs_full_mp[k] * pow_t
        pow_t = pow_t * t_mp
    return (mp_f(t_mp) - P) / t_mp**n


def auto_eps_analytic(coeffs_full, n, order, tol, safety=0.75):
    """Mirror of taylor_remainder.auto_eps -- relative-tolerance, scale-invariant."""
    v_lead, k_lead = 0.0, n + order
    for k in range(n, n + order):
        v = abs(float(coeffs_full[k]))
        if v > 1e-300:
            v_lead, k_lead = v, k
            break
    if v_lead == 0.0:
        return 1.0
    v_trunc, k_trunc = 0.0, n + order
    for extra in range(5):
        v = abs(float(coeffs_full[n + order + extra]))
        if v > 1e-300:
            v_trunc, k_trunc = v, n + order + extra
            break
    if v_trunc == 0.0:
        return 1.0
    return safety * (tol * v_lead / v_trunc) ** (1.0 / (k_trunc - k_lead))


# -------- runtime measurement --------


def time_call_ns_per_elem(fn, t_array, n_warmup=2, n_trials=8):
    """Median of (best of n_trials) measurements, normalized per element."""
    for _ in range(n_warmup):
        fn(t_array)
    best = float("inf")
    for _ in range(n_trials):
        t0 = time.perf_counter_ns()
        fn(t_array)
        elapsed = time.perf_counter_ns() - t0
        if elapsed < best:
            best = elapsed
    return best / len(t_array)


def benchmark_runtime(coeffs_full, np_f, n, order, dtype, N=100_000):
    t = np.linspace(1e-3, 0.5, N).astype(dtype)

    cs_poly = np.array([float(c) for c in coeffs_full[n : n + order]], dtype=dtype)

    def poly_branch(t, cs=cs_poly):
        out = np.full_like(t, cs[-1])
        for c in cs[-2::-1]:
            out = out * t + c
        return out

    cs_pre = np.array([float(c) for c in coeffs_full[:n]], dtype=dtype)

    def closed_branch(t, np_f=np_f, cs_pre=cs_pre, n=n):
        f_t = np_f(t)
        if n == 0:
            return f_t
        P = np.full_like(t, cs_pre[-1]) if len(cs_pre) else np.zeros_like(t)
        for c in cs_pre[-2::-1]:
            P = P * t + c
        return (f_t - P) / t**n

    poly_ns = time_call_ns_per_elem(poly_branch, t)
    closed_ns = time_call_ns_per_elem(closed_branch, t)
    return poly_ns, closed_ns


# -------- per-(function, dtype, order) measurement --------


def measure_one(name, mp_f, np_f, coeffs_mp, n, order, dtype):
    len(coeffs_mp)
    coeffs_f64 = [float(c) for c in coeffs_mp]
    eps_machine = float(np.finfo(dtype).eps)
    tol = 10 * eps_machine

    eps_pred = auto_eps_analytic(coeffs_f64, n, order, tol, safety=0.75)

    # error sweep (a coarser sweep -- the headline numbers we want are
    # crossover + runtime)
    ts = np.logspace(-15, np.log10(0.5), 100)
    refs = np.array([float(reference_value(mp_f, coeffs_mp, t, n)) for t in ts])
    poly = eval_poly_dtype(coeffs_mp, ts, n, order, dtype).astype(np.float64)
    closed = eval_closed_dtype(np_f, coeffs_mp, ts, n, dtype).astype(np.float64)
    ref_safe = np.where(np.abs(refs) > 1e-300, np.abs(refs), 1.0)
    err_poly = np.abs(poly - refs) / ref_safe
    err_closed = np.abs(closed - refs) / ref_safe

    eps_emp = float("nan")
    for i in range(len(ts)):
        if err_closed[i] < err_poly[i]:
            eps_emp = ts[i]
            break

    below_tol = np.where(err_poly <= tol)[0]
    eps_poly_tol = ts[below_tol[-1]] if len(below_tol) else float("nan")

    poly_ns, closed_ns = benchmark_runtime(coeffs_mp, np_f, n, order, dtype)

    return {
        "name": name,
        "n": n,
        "order": order,
        "dtype": dtype.__name__,
        "eps_pred": eps_pred,
        "eps_emp": eps_emp,
        "eps_poly_tol": eps_poly_tol,
        "poly_ns": poly_ns,
        "closed_ns": closed_ns,
    }


def main():
    cases = [
        ("log1p(x)/x       ", mp.log1p, np.log1p, coeffs_log1p_mp(40), 1),
        (
            "expm1(x)/x       ",
            lambda x: mp.exp(x) - 1,
            np.expm1,
            coeffs_expm1_mp(40),
            1,
        ),
        (
            "(cos x - 1)/x^2  ",
            lambda x: mp.cos(x) - 1,
            lambda x: np.cos(x) - 1,
            coeffs_cosm1_mp(40),
            2,
        ),
    ]

    orders = [4, 6, 8, 10, 12, 14]
    dtypes = [np.float32, np.float64]

    header = (
        f"{'function':<18} {'dtype':<8} {'order':>5}  "
        f"{'eps_pred':>10} {'eps_emp':>10} {'poly_below_tol':>14}  "
        f"{'poly_ns':>8} {'closed_ns':>10} {'poly/closed':>11}"
    )
    print(header)
    print("-" * len(header))

    for name, mp_f, np_f, coeffs_mp, n in cases:
        for dtype in dtypes:
            for order in orders:
                if n + order >= len(coeffs_mp):
                    continue
                r = measure_one(name, mp_f, np_f, coeffs_mp, n, order, dtype)
                ratio = r["poly_ns"] / r["closed_ns"]
                emp = f"{r['eps_emp']:.3g}" if not math.isnan(r["eps_emp"]) else "  --"
                below = (
                    f"{r['eps_poly_tol']:.3g}"
                    if not math.isnan(r["eps_poly_tol"])
                    else "  --"
                )
                print(
                    f"{r['name']:<18} {r['dtype']:<8} {r['order']:>5}  "
                    f"{r['eps_pred']:>10.3g} {emp:>10} {below:>14}  "
                    f"{r['poly_ns']:>7.1f} {r['closed_ns']:>9.1f} {ratio:>10.2f}x"
                )
            print()


if __name__ == "__main__":
    main()
