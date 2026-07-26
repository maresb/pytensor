"""Plots from the taylor_remainder branch experiments.

Produces three figures saved as PNG in dev/taylor/:

  fig1_error_vs_t.png   error of each branch as a function of |t|, with
                        auto_eps marker, per (function, dtype)
  fig2_eps_vs_order.png largest |t| at which poly stays within 10·eps_machine,
                        as a function of order, per (function, dtype)
  fig3_runtime.png      per-element runtime of poly vs closed branch as
                        a function of order, per (function, dtype)
"""

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


mp.mp.dps = 50
HERE = Path(__file__).resolve().parent


# -------- coefficients --------


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


CASES = [
    {
        "name": "log1p(x)/x",
        "mp": mp.log1p,
        "np": np.log1p,
        "coeffs": coeffs_log1p_mp(40),
        "n": 1,
    },
    {
        "name": "expm1(x)/x",
        "mp": lambda x: mp.exp(x) - 1,
        "np": np.expm1,
        "coeffs": coeffs_expm1_mp(40),
        "n": 1,
    },
    {
        "name": "(cos(x)-1)/x^2",
        "mp": lambda x: mp.cos(x) - 1,
        "np": lambda x: np.cos(x) - 1,
        "coeffs": coeffs_cosm1_mp(40),
        "n": 2,
    },
]


# -------- evaluators --------


def eval_poly(coeffs_full, t, n, order, dtype):
    cs = np.array([float(c) for c in coeffs_full[n : n + order]], dtype=dtype)
    t = np.asarray(t, dtype=dtype)
    out = np.full_like(t, cs[-1])
    for c in cs[-2::-1]:
        out = out * t + c
    return out


def eval_closed(np_f, coeffs_full, t, n, dtype):
    t = np.asarray(t, dtype=dtype)
    f_t = np_f(t)
    if n == 0:
        return f_t
    cs = np.array([float(c) for c in coeffs_full[:n]], dtype=dtype)
    P = np.full_like(t, cs[-1])
    for c in cs[-2::-1]:
        P = P * t + c
    return (f_t - P) / t**n


def reference(mp_f, coeffs_mp, t, n):
    t_mp = mp.mpf(float(t))
    if n == 0:
        return mp_f(t_mp)
    P = mp.mpf(0)
    pow_t = mp.mpf(1)
    for k in range(n):
        P = P + coeffs_mp[k] * pow_t
        pow_t = pow_t * t_mp
    return (mp_f(t_mp) - P) / t_mp**n


def auto_eps_analytic(coeffs_full, n, order, tol, safety=0.75, max_extra=4):
    for extra in range(max_extra + 1):
        v = abs(float(coeffs_full[n + order + extra]))
        if v > 1e-300:
            return safety * (tol / v) ** (1.0 / (order + extra))
    return 1.0


# -------- error sweep --------


def error_curves(case, order, dtype):
    coeffs_mp = case["coeffs"]
    n = case["n"]
    ts = np.logspace(-15, np.log10(0.5), 200)
    refs = np.array([float(reference(case["mp"], coeffs_mp, t, n)) for t in ts])
    poly = eval_poly(coeffs_mp, ts, n, order, dtype).astype(np.float64)
    closed = eval_closed(case["np"], coeffs_mp, ts, n, dtype).astype(np.float64)
    ref_safe = np.where(np.abs(refs) > 1e-300, np.abs(refs), 1.0)
    err_poly = np.abs(poly - refs) / ref_safe
    err_closed = np.abs(closed - refs) / ref_safe
    return ts, err_poly, err_closed


def largest_t_below_tol(ts, err_poly, tol):
    below = np.where(err_poly <= tol)[0]
    return ts[below[-1]] if len(below) else float("nan")


# -------- runtime --------


def time_per_elem(fn, t_array, n_warmup=2, n_trials=10):
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


def benchmark(case, order, dtype, N=100_000):
    coeffs_mp = case["coeffs"]
    n = case["n"]
    t = np.linspace(1e-3, 0.5, N).astype(dtype)

    cs_poly = np.array([float(c) for c in coeffs_mp[n : n + order]], dtype=dtype)

    def poly_branch(t, cs=cs_poly):
        out = np.full_like(t, cs[-1])
        for c in cs[-2::-1]:
            out = out * t + c
        return out

    cs_pre = np.array([float(c) for c in coeffs_mp[:n]], dtype=dtype)

    def closed_branch(t, np_f=case["np"], cs_pre=cs_pre, n=n):
        f_t = np_f(t)
        if n == 0:
            return f_t
        P = np.full_like(t, cs_pre[-1]) if len(cs_pre) else np.zeros_like(t)
        for c in cs_pre[-2::-1]:
            P = P * t + c
        return (f_t - P) / t**n

    return time_per_elem(poly_branch, t), time_per_elem(closed_branch, t)


# -------- figures --------


def fig1_error_vs_t():
    """Per-function: error of poly and closed vs |t|, with eps markers."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    order = 10
    dtypes = [np.float64, np.float32]
    for col, case in enumerate(CASES):
        coeffs_f64 = [float(c) for c in case["coeffs"]]
        for row, dtype in enumerate(dtypes):
            ax = axes[row, col]
            eps_machine = float(np.finfo(dtype).eps)
            tol = 10 * eps_machine
            ts, err_poly, err_closed = error_curves(case, order, dtype)
            eps_pred = auto_eps_analytic(coeffs_f64, case["n"], order, tol)
            t_below = largest_t_below_tol(ts, err_poly, tol)

            ax.loglog(
                ts, np.maximum(err_poly, 1e-20), label="poly branch", color="C0", lw=1.6
            )
            ax.loglog(
                ts,
                np.maximum(err_closed, 1e-20),
                label="closed branch",
                color="C1",
                lw=1.6,
            )
            ax.axvline(
                eps_pred,
                color="k",
                linestyle="--",
                alpha=0.6,
                label=f"auto_eps={eps_pred:.2g}",
            )
            if not math.isnan(t_below):
                ax.axvline(
                    t_below,
                    color="C2",
                    linestyle=":",
                    alpha=0.7,
                    label=f"poly<tol up to {t_below:.2g}",
                )
            ax.axhline(tol, color="gray", linestyle="-", alpha=0.4, lw=0.8)
            ax.set_xlabel("|t|")
            if col == 0:
                ax.set_ylabel(f"relative error  ({dtype.__name__})")
            ax.set_title(f"{case['name']}, n={case['n']}, order={order}")
            ax.set_ylim(1e-18, 1e2)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"branch error vs |t|  (order=10, ref=mpmath@{mp.mp.dps}d)", fontsize=13
    )
    fig.tight_layout()
    out = HERE / "fig1_error_vs_t.png"
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


def fig2_eps_vs_order():
    """Largest |t| at which poly stays within 10·eps_machine, vs order."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    orders = list(range(4, 22, 2))
    dtypes = [np.float64, np.float32]

    for ax, dtype in zip(axes, dtypes):
        eps_machine = float(np.finfo(dtype).eps)
        tol = 10 * eps_machine
        for case in CASES:
            coeffs_f64 = [float(c) for c in case["coeffs"]]
            ys_emp, ys_pred = [], []
            for order in orders:
                if case["n"] + order >= len(case["coeffs"]):
                    ys_emp.append(np.nan)
                    ys_pred.append(np.nan)
                    continue
                ts, err_poly, _ = error_curves(case, order, dtype)
                ys_emp.append(largest_t_below_tol(ts, err_poly, tol))
                ys_pred.append(auto_eps_analytic(coeffs_f64, case["n"], order, tol))
            ax.semilogy(
                orders, ys_emp, marker="o", label=f"{case['name']} empirical", lw=1.7
            )
            ax.semilogy(
                orders,
                ys_pred,
                marker="x",
                linestyle="--",
                label=f"{case['name']} auto_eps",
                lw=1,
                alpha=0.7,
                color=ax.lines[-1].get_color(),
            )
        ax.set_xlabel("polynomial order")
        ax.set_ylabel("largest |t| with poly error <= 10·eps_machine")
        ax.set_title(f"dtype={dtype.__name__}  (eps_machine={eps_machine:.1e})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("polynomial branch domain coverage vs order", fontsize=13)
    fig.tight_layout()
    out = HERE / "fig2_eps_vs_order.png"
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


def fig3_runtime():
    """Per-element runtime of poly and closed branch vs order."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    orders = list(range(4, 22, 2))
    dtypes = [np.float64, np.float32]

    for ax, dtype in zip(axes, dtypes):
        for case in CASES:
            poly_ns_list = []
            closed_ns_list = []
            for order in orders:
                if case["n"] + order >= len(case["coeffs"]):
                    poly_ns_list.append(np.nan)
                    closed_ns_list.append(np.nan)
                    continue
                p_ns, c_ns = benchmark(case, order, dtype)
                poly_ns_list.append(p_ns)
                closed_ns_list.append(c_ns)
            (line,) = ax.plot(
                orders, poly_ns_list, marker="o", label=f"{case['name']} poly", lw=1.7
            )
            ax.plot(
                orders,
                closed_ns_list,
                marker="s",
                linestyle="--",
                label=f"{case['name']} closed",
                alpha=0.7,
                color=line.get_color(),
            )
        ax.set_xlabel("polynomial order")
        ax.set_ylabel("ns / element")
        ax.set_title(f"dtype={dtype.__name__}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("per-element runtime: polynomial vs closed branch", fontsize=13)
    fig.tight_layout()
    out = HERE / "fig3_runtime.png"
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


def main():
    print("generating figures...")
    fig1_error_vs_t()
    fig2_eps_vs_order()
    fig3_runtime()


if __name__ == "__main__":
    main()
