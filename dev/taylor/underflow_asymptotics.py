"""Asymptotic error measurement for naive f(x)/x with f(x) ~ K·x near 0.

For each K, sweep |x| logarithmically and compute the relative error of
naive (K·sin(x))/x against an mpmath reference. Also plot/print the predicted
boundaries:

  underflow-zero boundary:    |x| ~ smallest_subnormal / |K|
  subnormal-precision boundary: |x| ~ smallest_normal / |K|

Empirical question: does the error follow the predicted
  rel_err ~ min(1, smallest_subnormal / (|K| · |x|))   for K·x in subnormal range
"""

import math

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


mp.mp.dps = 50

# dtype constants (e_min - p gives smallest subnormal exponent)
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
    )


SMALLEST_NORMAL = float(np.finfo(np.float64).smallest_normal)
SMALLEST_SUBNORMAL = math.ldexp(1.0, -1074)
EPS_MACHINE = float(np.finfo(np.float64).eps)


def naive(K, x, dtype=np.float64):
    K_ = np.asarray(K, dtype=dtype)
    x_ = np.asarray(x, dtype=dtype)
    return (K_ * np.sin(x_)) / x_


def reference(K, x):
    return float(mp.mpf(K) * mp.sin(mp.mpf(x)) / mp.mpf(x))


def sweep_K(K, log_x_range=(-323, 0), n=400):
    log_xs = np.linspace(log_x_range[0], log_x_range[1], n)
    xs = 10.0**log_xs
    refs = np.array([reference(K, x) for x in xs])
    with np.errstate(all="ignore"):
        vals = np.array([naive(K, x) for x in xs])
    # true relative error
    safe = np.where(np.abs(refs) > 0, np.abs(refs), 1.0)
    rel = np.abs(vals - refs) / safe
    rel[~np.isfinite(rel)] = 1.0
    return xs, rel


def predicted_rel_err(K, x):
    """Theoretical model:
      regime A (|K·x| >= smallest_normal):   ~eps_machine
      regime B (subnormal):                  smallest_subnormal / (|K| · |x|)
      regime C (|K·x| < smallest_subnormal): 1
    Values of K and x not exactly equal to K·x; this approximation uses the
    leading-order behavior f(x) ~ K·x.
    """
    Kx = abs(K * x)
    if Kx >= SMALLEST_NORMAL:
        return EPS_MACHINE
    if Kx >= SMALLEST_SUBNORMAL:
        return min(1.0, SMALLEST_SUBNORMAL / Kx)
    return 1.0


def sweep_K_around_boundary(K, n=200):
    """Sample more densely in the [smallest_subnormal/K, smallest_normal/K]
    transition zone so we can see the subnormal-degradation regime.

    Clamps below smallest_subnormal (otherwise float underflows to zero
    and the reference computation divides by zero).
    """
    ub_log = math.log10(SMALLEST_SUBNORMAL / K)
    sb_log = math.log10(SMALLEST_NORMAL / K)
    floor_log = math.log10(SMALLEST_SUBNORMAL) + 0.5
    below_lo = max(floor_log, ub_log - 4)
    log_xs = np.concatenate(
        [
            np.linspace(below_lo, ub_log, 30),
            np.linspace(ub_log, sb_log, max(2, n - 80)),
            np.linspace(sb_log, sb_log + 4, 50),
        ]
    )
    xs = 10.0**log_xs
    refs = np.array([reference(K, x) for x in xs])
    with np.errstate(all="ignore"):
        vals = np.array([naive(K, x) for x in xs])
    safe = np.where(np.abs(refs) > 0, np.abs(refs), 1.0)
    rel = np.abs(vals - refs) / safe
    rel[~np.isfinite(rel)] = 1.0
    return xs, rel


def sweep_around_boundary_dtype(K, dtype, n=200):
    consts = dtype_constants(dtype)
    sn = consts["smallest_normal"]
    ss = consts["smallest_subnormal"]
    eps = consts["eps_machine"]

    K_dtype = float(np.asarray(K, dtype=dtype))
    if K_dtype == 0:
        return None  # K underflowed in this dtype
    ub_log = math.log10(ss / abs(K_dtype))
    sb_log = math.log10(sn / abs(K_dtype))
    floor_log = math.log10(ss) + 0.5
    below_lo = max(floor_log, ub_log - 4)
    log_xs = np.concatenate(
        [
            np.linspace(below_lo, ub_log, 30),
            np.linspace(ub_log, sb_log, max(2, n - 80)),
            np.linspace(sb_log, sb_log + 4, 50),
        ]
    )
    xs = 10.0**log_xs
    refs = np.array([reference(K, x) for x in xs])
    with np.errstate(all="ignore"):
        vals = np.array([float(naive(K, x, dtype)) for x in xs])
    safe = np.where(np.abs(refs) > 0, np.abs(refs), 1.0)
    rel = np.abs(vals - refs) / safe
    rel[~np.isfinite(rel)] = 1.0
    return xs, rel, ub_log, sb_log, eps, sn, ss


def main_multi_dtype():
    """Verify the asymptotic formula for float32 and float16 too."""
    cases = [
        (np.float64, [1.0, 1e-100, 1e-300]),
        (np.float32, [1.0, 1e-20, 1e-40]),
        (np.float16, [1.0, 1e-3, 1e-5]),
    ]
    fig, axes = plt.subplots(
        len(cases), 3, figsize=(12, 4 * len(cases)), sharex=False, sharey=True
    )
    for row, (dtype, Ks) in enumerate(cases):
        for col, K in enumerate(Ks):
            ax = axes[row, col]
            sweep = sweep_around_boundary_dtype(K, dtype)
            if sweep is None:
                ax.text(
                    0.5,
                    0.5,
                    f"K={K:g} underflows in {dtype.__name__}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue
            xs, rel, ub_log, sb_log, eps_m, _sn, ss = sweep
            ax.loglog(
                xs, np.maximum(rel, 1e-20), "C0o-", lw=1.0, ms=2, label="empirical"
            )
            pred = np.array([min(1.0, max(eps_m, ss / abs(K * x))) for x in xs])
            ax.loglog(xs, pred, "k--", lw=1.0, alpha=0.7, label="model")
            ax.axvline(10**ub_log, color="C3", linestyle=":", lw=1.0)
            ax.axvline(10**sb_log, color="C2", linestyle=":", lw=1.0)
            ax.axhline(eps_m, color="gray", linestyle="-", lw=0.5, alpha=0.5)
            ax.set_xlabel("|x|")
            if col == 0:
                ax.set_ylabel(f"rel_err  ({dtype.__name__})")
            ax.set_title(f"{dtype.__name__}  K={K:g}")
            ax.set_ylim(eps_m / 10, 10)
            ax.legend(fontsize=7, loc="lower left")
            ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Naive (K·sin(x))/x error  --  asymptotic formula validates "
        "across float64 / float32 / float16",
        fontsize=11,
    )
    fig.tight_layout()
    out = "underflow_asymptotics_dtypes.png"
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


def main():
    main_multi_dtype()
    Ks = [1.0, 1e-50, 1e-100, 1e-200, 1e-300]

    fig, axes = plt.subplots(1, len(Ks), figsize=(4 * len(Ks), 4), sharey=True)
    for ax, K in zip(axes, Ks):
        xs, rel = sweep_K_around_boundary(K)
        ax.loglog(xs, np.maximum(rel, 1e-20), "C0o-", lw=1.0, ms=2, label="empirical")
        # model:  rel ~ smallest_subnormal/(|K·x|)  in the subnormal regime
        pred = np.array(
            [min(1.0, max(EPS_MACHINE, SMALLEST_SUBNORMAL / abs(K * x))) for x in xs]
        )
        ax.loglog(xs, pred, "k--", lw=1.0, alpha=0.7, label="model: 2^-1074/(K|x|)")
        # boundaries
        ub = SMALLEST_SUBNORMAL / K
        sb = SMALLEST_NORMAL / K
        ax.axvline(
            ub, color="C3", linestyle=":", lw=1.0, label=f"underflow: |x|={ub:.1e}"
        )
        ax.axvline(
            sb, color="C2", linestyle=":", lw=1.0, label=f"subnormal: |x|={sb:.1e}"
        )
        ax.axhline(EPS_MACHINE, color="gray", linestyle="-", lw=0.5, alpha=0.5)
        ax.set_xlabel("|x|")
        ax.set_title(f"K = {K:g}")
        ax.set_ylim(EPS_MACHINE / 10, 10)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("relative error of (K·sin(x))/x")
    fig.suptitle(
        "Naive  (K · sin(x)) / x  --  empirical error follows  "
        "rel_err ≈ min(1, 2^-1074/(K·|x|))  in the subnormal regime",
        fontsize=11,
    )
    fig.tight_layout()
    out = "underflow_asymptotics.png"
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")

    # Also produce a tabular summary
    print(f"\n  smallest_normal    = {SMALLEST_NORMAL:.3e}")
    print(f"  smallest_subnormal = {SMALLEST_SUBNORMAL:.3e}")
    print(f"  eps_machine        = {EPS_MACHINE:.3e}\n")
    print(
        f"  {'K':>10}  {'|x| underflow boundary':>22}  "
        f"{'|x| subnormal boundary':>22}  {'safe |x| floor':>15}"
    )
    print("  " + "-" * 80)
    for K in Ks:
        ub = SMALLEST_SUBNORMAL / K
        sb = SMALLEST_NORMAL / K
        # round-trip the predicted boundaries: (smallest_subnormal/K) is the |x|
        # below which naive is wrong; we'd want polynomial branch to cover at
        # least up to ub.
        print(f"  {K:>10.0e}  {ub:>22.3e}  {sb:>22.3e}  {ub:>15.3e}")


if __name__ == "__main__":
    main()
