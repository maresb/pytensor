"""Empirical saturation audit: across the test suite, find the
worst-case (max obs/bound) ratio for each bound. A ratio near 1
means the bound is tight; a ratio << 1 means the bound is loose.
"""

import math
import sys
import warnings

import mpmath as mp


sys.path.insert(0, "/home/mares/repos/pytensor/dev/taylor")
from taylor_remainder import (
    TaylorAtPoint,
    TaylorRemainderClosedCancellationWarning,
    auto_eps,
    closed_branch_rel_err_bound,
    poly_branch_rel_err_bound,
    taylor_remainder,
)

import pytensor
import pytensor.tensor as pt


mp.mp.dps = 60
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.on_opt_error = "ignore"
warnings.simplefilter("ignore", TaylorRemainderClosedCancellationWarning)

records = []


def measure(family, K, n, order, v, fn, ref_fn, cache):
    eps = auto_eps(cache, n=n, order=order)
    out = float(fn(v))
    ref = float(ref_fn(K, mp.mpf(v)))
    if not math.isfinite(out):
        return
    ref_safe = abs(ref) if abs(ref) > 1e-300 else 1.0
    rel = abs(out - ref) / ref_safe
    if v < eps:
        bound = poly_branch_rel_err_bound(order=order, cache=cache, n=n, v=v)
        branch = "poly"
    else:
        bound = closed_branch_rel_err_bound(cache, n=n, v=v, order=order)
        branch = "closed"
    if bound == float("inf") or bound == 0:
        return
    ratio = rel / bound
    records.append(
        {
            "family": family,
            "K": K,
            "n": n,
            "order": order,
            "v": v,
            "branch": branch,
            "rel_err": rel,
            "bound": bound,
            "ratio": ratio,
        }
    )


# ---- fixture: cos(K*x), n=2 ------------------------------------------------
def ref_cos_n2(K, x):
    if x == 0:
        return -(mp.mpf(K) ** 2) / 2
    return (mp.cos(mp.mpf(K) * x) - 1) / x**2


for K in [0.1, 1.0, 10.0, 100.0]:
    x = pt.dscalar("x")
    f = pt.cos(K * x)
    cache = TaylorAtPoint(f, x, 0.0)
    eps_v = auto_eps(cache, n=2, order=10)
    fn = pytensor.function([x], taylor_remainder(f, x, 0.0, 2, order=10))
    for frac in [1e-6, 0.1, 0.5, 0.9, 0.99, 1.0, 1.001, 1.01, 1.1, 1.5, 2.0, 5.0]:
        v = eps_v * frac
        if v <= 0.5 and v != 0:
            measure("cos_Kx_n2", K, 2, 10, v, fn, ref_cos_n2, cache)


# ---- fixture: sin(K*x), n=3 ------------------------------------------------
def ref_sin_n3(K, x):
    if x == 0:
        return -(mp.mpf(K) ** 3) / 6
    Kx = mp.mpf(K) * x
    return (mp.sin(Kx) - Kx) / x**3


for K in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    x = pt.dscalar("x")
    f = pt.sin(K * x)
    cache = TaylorAtPoint(f, x, 0.0)
    eps_v = auto_eps(cache, n=3, order=10)
    fn = pytensor.function([x], taylor_remainder(f, x, 0.0, 3, order=10))
    for frac in [1e-6, 0.1, 0.5, 0.9, 0.99, 1.0, 1.001, 1.01, 1.1, 1.5, 2.0, 5.0]:
        v = eps_v * frac
        if v <= 0.5 and v != 0:
            measure("sin_Kx_n3", K, 3, 10, v, fn, ref_sin_n3, cache)


# ---- fixture: cos(K*x), n=3 (k_lead > n) -----------------------------------
def ref_cos_n3(K, x):
    if x == 0:
        return mp.mpf(0)
    Kx = mp.mpf(K) * x
    return (mp.cos(Kx) - 1 + Kx**2 / 2) / x**3


for K in [0.1, 1.0, 10.0, 100.0]:
    x = pt.dscalar("x")
    f = pt.cos(K * x)
    cache = TaylorAtPoint(f, x, 0.0)
    eps_v = auto_eps(cache, n=3, order=10)
    fn = pytensor.function([x], taylor_remainder(f, x, 0.0, 3, order=10))
    for frac in [1e-6, 0.1, 0.5, 0.9, 0.99, 1.0, 1.001, 1.01, 1.1, 1.5, 2.0, 5.0]:
        v = eps_v * frac
        if v <= 0.5 and v != 0:
            measure("cos_Kx_n3", K, 3, 10, v, fn, ref_cos_n3, cache)


# ---- fixture: sin(K*x) at n=2 ---------------------------------------------
def ref_sin_n2(K, x):
    if x == 0:
        return mp.mpf(K)
    return mp.sin(mp.mpf(K) * x) / x


for K in [0.1, 1.0, 10.0, 100.0]:
    x = pt.dscalar("x")
    f = pt.sin(K * x)
    cache = TaylorAtPoint(f, x, 0.0)
    eps_v = auto_eps(cache, n=1, order=10)
    fn = pytensor.function([x], taylor_remainder(f, x, 0.0, 1, order=10))
    for frac in [1e-6, 0.1, 0.5, 0.9, 0.99, 1.0, 1.001, 1.01, 1.1, 1.5, 2.0, 5.0]:
        v = eps_v * frac
        if v <= 0.5 and v != 0:
            measure("sin_Kx_n1", K, 1, 10, v, fn, ref_sin_n2, cache)


# ---- summarize -------------------------------------------------------------

print(f"\nTotal records: {len(records)}\n")

by_family_branch = {}
for r in records:
    key = (r["family"], r["branch"])
    by_family_branch.setdefault(key, []).append(r)

print(
    f"{'family':<14} {'branch':<7} {'n':>4} {'count':>5} {'max ratio':>10} {'mean ratio':>11} {'worst case':>40}"
)
print("-" * 110)
for (family, branch), recs in sorted(by_family_branch.items()):
    ratios = [r["ratio"] for r in recs]
    max_r = max(ratios)
    mean_r = sum(ratios) / len(ratios)
    worst = max(recs, key=lambda r: r["ratio"])
    worst_desc = f"K={worst['K']} v={worst['v']:.3g} rel={worst['rel_err']:.2e} b={worst['bound']:.2e}"
    print(
        f"{family:<14} {branch:<7} {worst['n']:>4} {len(recs):>5} {max_r:>10.3f} {mean_r:>11.3f} {worst_desc:>40}"
    )

print(f"\nMax obs/bound across ALL records: {max(r['ratio'] for r in records):.3f}")
print(f"Records with obs/bound > 0.5: {sum(1 for r in records if r['ratio'] > 0.5)}")
print(f"Records with obs/bound > 0.9: {sum(1 for r in records if r['ratio'] > 0.9)}")
