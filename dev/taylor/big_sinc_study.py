"""f(x) = K · sin(x) -- scale invariance of auto_eps and taylor_remainder.

Tests both K = 1e50 (huge) and K = 1e-50 (tiny). The relative error of any
reasonable evaluation of R(x) = K · sin(x) / x should be K-invariant.
"""

import math

import mpmath as mp
from taylor_remainder import TaylorAtPoint, auto_eps, taylor_remainder

import pytensor
import pytensor.tensor as pt


mp.mp.dps = 50


def run_for_K(K):
    print(f"\n=========== K = {K:g} ===========")
    x = pt.dscalar("x")
    f = K * pt.sin(x)

    cache = TaylorAtPoint(f, x, 0.0)
    eps_chosen = auto_eps(cache, n=1, order=10)
    cache_unscaled = TaylorAtPoint(pt.sin(x), x, 0.0)
    eps_unscaled = auto_eps(cache_unscaled, n=1, order=10)
    print(f"auto_eps for K·sin(x):   {eps_chosen:.4g}")
    print(f"auto_eps for plain sin:  {eps_unscaled:.4g}")
    print(f"ratio:                   {eps_chosen / eps_unscaled:.4g}  (should be 1)")

    def reference(x_):
        if x_ == 0:
            return float(mp.mpf(K))
        x_mp = mp.mpf(float(x_))
        return float(mp.mpf(K) * mp.sin(x_mp) / x_mp)

    y = taylor_remainder(f, x, 0.0, 1, order=10)
    fn_taylor = pytensor.function([x], y)
    fn_naive = pytensor.function([x], f / x)

    print(
        f"\n{'x':>10} {'taylor_remainder':>22} {'naive K·sin(x)/x':>22} "
        f"{'reference':>22} {'tr_rel':>11} {'naive_rel':>11}"
    )
    for v in [0.0, 1e-12, 1e-4, 1e-2, eps_chosen * 0.5, eps_chosen * 2, 0.1, 1.0]:
        ref = reference(v)
        try:
            tr = float(fn_taylor(v))
        except Exception:
            tr = float("nan")
        try:
            naive = float(fn_naive(v))
        except Exception:
            naive = float("nan")
        ref_safe = abs(ref) if ref != 0 else 1.0
        tr_err = abs(tr - ref) / ref_safe if math.isfinite(tr) else float("inf")
        naive_err = (
            abs(naive - ref) / ref_safe if math.isfinite(naive) else float("inf")
        )
        print(
            f"{v:>10.2e} {tr:>22.12g} {naive:>22.12g} {ref:>22.12g} "
            f"{tr_err:>11.2e} {naive_err:>11.2e}"
        )


def main():
    run_for_K(1e50)
    run_for_K(1e-50)
    run_for_K(1.0)


if __name__ == "__main__":
    main()
