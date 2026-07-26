"""Asymptotic error of naive f(x)/x for f(x) = K·x·(1 + L·x).

f vanishes to order 1 with c_1 = K, c_2 = K·L. The leading behavior near x=0
is f(x) ~ K·x, so the underflow asymptotic should depend only on K (not L).

Goal: verify that
    rel_err(naive f(x)/x)  ≈  min(1, 2^{e_min - p} / (|K| · |x|))
holds across the full representable range of K (and dtype), independent of L.
"""

import math

import mpmath as mp
import numpy as np


mp.mp.dps = 50

DTYPE_INFO = {
    np.float16: dict(e_min=-14, p=10),
    np.float32: dict(e_min=-126, p=23),
    np.float64: dict(e_min=-1022, p=52),
}


def dtype_constants(dtype):
    info = DTYPE_INFO[dtype]
    return dict(
        smallest_normal=math.ldexp(1.0, info["e_min"]),
        smallest_subnormal=math.ldexp(1.0, info["e_min"] - info["p"]),
        eps_machine=math.ldexp(1.0, -info["p"]),
        largest=float(np.finfo(dtype).max),
    )


def naive(K, L, x, dtype):
    K_ = np.asarray(K, dtype=dtype)
    L_ = np.asarray(L, dtype=dtype)
    x_ = np.asarray(x, dtype=dtype)
    return float((K_ * x_ * (1 + L_ * x_)) / x_)


def reference(K, L, x):
    """High-precision (K · x · (1 + L·x)) / x  =  K·(1 + L·x)."""
    return float(mp.mpf(K) * (mp.mpf(1) + mp.mpf(L) * mp.mpf(x)))


def predicted(K, L, x, dtype):
    consts = dtype_constants(dtype)
    c1 = abs(K)  # f'(0) = K; L doesn't enter the underflow asymptotic
    if c1 == 0:
        return 0.0
    return min(
        1.0, max(consts["eps_machine"], consts["smallest_subnormal"] / (c1 * abs(x)))
    )


def sample_x(dtype, n_per_decade=4):
    consts = dtype_constants(dtype)
    log_lo = math.log10(consts["smallest_subnormal"]) + 0.5
    log_hi = 0.0  # |x| up to 1
    log_xs = np.linspace(log_lo, log_hi, int(n_per_decade * (log_hi - log_lo)))
    return 10.0**log_xs


def sample_K_full_range(dtype, n_per_decade=2):
    """Cover the full representable K range, from smallest subnormal to near max."""
    consts = dtype_constants(dtype)
    log_lo = math.log10(consts["smallest_subnormal"]) + 0.5
    log_hi = math.log10(consts["largest"]) - 0.5
    log_Ks = np.linspace(log_lo, log_hi, max(2, int(n_per_decade * (log_hi - log_lo))))
    return 10.0**log_Ks


def run_dtype(dtype, Ls=(0.0, 1.0, -1.0, 10.0, 1000.0, -1e6)):
    Ks = sample_K_full_range(dtype)
    xs = sample_x(dtype)

    print(
        f"\n=== dtype={dtype.__name__}  "
        f"K range: [{Ks[0]:.2e}, {Ks[-1]:.2e}] ({len(Ks)} pts)  "
        f"|x| range: [{xs[0]:.2e}, {xs[-1]:.2e}] ({len(xs)} pts) ==="
    )

    for L in Ls:
        # tally agreement with the predicted formula across (K, x)
        n_pts, n_match, max_excess = 0, 0, 0.0
        worst = None
        for K in Ks:
            K_dtype = float(np.asarray(K, dtype=dtype))
            if K_dtype == 0:
                continue
            for x in xs:
                ref = reference(K, L, x)
                with np.errstate(all="ignore"):
                    val = naive(K, L, x, dtype)
                if ref == 0.0:
                    rel = abs(val)
                else:
                    if not math.isfinite(val):
                        rel = float("inf")
                    else:
                        rel = abs(val - ref) / abs(ref)
                pred = predicted(K, L, x, dtype)
                n_pts += 1
                # "match" means empirical <= 10·pred (within an order of magnitude)
                if rel <= 10 * pred:
                    n_match += 1
                else:
                    excess = rel / max(pred, 1e-300)
                    if excess > max_excess:
                        max_excess = excess
                        worst = (K, x, ref, val, rel, pred)
        print(
            f"  L={L:>10.4g}  match: {n_match}/{n_pts}  max excess: {max_excess:.2g}",
            end="",
        )
        if worst is not None and max_excess > 100:
            K, x, ref, val, rel, pred = worst
            print(
                f"   worst: K={K:.2e} x={x:.2e} got={val:.2e} ref={ref:.2e} "
                f"rel={rel:.2e} pred={pred:.2e}"
            )
        else:
            print()


def main():
    for dtype in (np.float64, np.float32, np.float16):
        run_dtype(dtype)


if __name__ == "__main__":
    main()
