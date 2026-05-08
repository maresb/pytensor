"""sinc(x) = sin(x)/x  -- division-only error case study.

For sinc:
  - n = 1
  - f(x) = sin(x), which vanishes to order 1 at a=0
  - so P_0 = 0 and the closed branch is f(x)/(x-a)^n = sin(x)/x with NO
    subtraction, just one division
  - all error comes from how sin(x)/x behaves in floating point

Question: for which |x| does the naive sin(x)/x lose precision?

Reference: mpmath at 50 digits.
Test: numpy float64 and float32 evaluations of sin(x)/x.
"""

import mpmath as mp
import numpy as np


mp.mp.dps = 50


def naive_sinc(x, dtype):
    x = np.asarray(x, dtype=dtype)
    return np.sin(x) / x


def reference_sinc(x):
    """sin(x)/x at high precision, with the limit value 1 at x=0."""
    if x == 0:
        return 1.0
    return float(mp.sin(mp.mpf(float(x))) / mp.mpf(float(x)))


def main():
    # Sweep limited to "values representable as a normal float in both dtypes".
    # Below smallest_normal_f32 (~1.18e-38) float32 can't represent x non-zero,
    # so any naive failure is just underflow-to-zero, not a sinc-specific issue.
    ts_pos = np.logspace(-37, 1, 200)
    ts = np.concatenate([ts_pos, [0.0]])
    ts.sort()

    refs = np.array([reference_sinc(t) for t in ts])
    refs_safe = np.where(np.abs(refs) > 1e-300, np.abs(refs), 1.0)

    for dtype in (np.float64, np.float32):
        eps_machine = float(np.finfo(dtype).eps)
        smallest_normal = float(np.finfo(dtype).smallest_normal)
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = naive_sinc(ts, dtype).astype(np.float64)

        # mark NaN/Inf
        bad = ~np.isfinite(vals)
        rel_err = np.abs(vals - refs) / refs_safe
        rel_err[bad] = np.inf

        # find regions of trouble
        threshold = 10 * eps_machine
        bad_idx = np.where(rel_err > threshold)[0]

        print(
            f"\n=== dtype={dtype.__name__}  "
            f"eps_machine={eps_machine:.2e}  "
            f"smallest_normal={smallest_normal:.2e} ==="
        )

        if len(bad_idx) == 0:
            print(
                f"  naive sin(x)/x is accurate to <= 10·eps_machine for all "
                f"sampled |x| in [{ts[0]:.0e}, {ts[-1]:.0e}]"
            )
        else:
            print(
                f"  naive sin(x)/x exceeds 10·eps_machine at {len(bad_idx)} "
                f"sampled points"
            )
            # contiguous "bad" intervals
            print("  trouble regions:")
            t_bad = ts[bad_idx]
            r_bad = rel_err[bad_idx]
            v_bad = vals[bad_idx]
            for t, r, v in zip(t_bad[:5], r_bad[:5], v_bad[:5]):
                marker = "NaN" if not np.isfinite(v) else f"val={v:.6g}"
                print(f"    |x|={t:>10.2e}  rel_err={r:.2e}  {marker}")
            if len(t_bad) > 5:
                print(f"    ...({len(t_bad) - 5} more)")

        # specifically check x = 0
        x0 = np.zeros((), dtype=dtype)
        with np.errstate(invalid="ignore", divide="ignore"):
            v0 = float(naive_sinc(x0, dtype))
        print(f"  naive at x=0 exactly: {v0}")

        # look at x where naive disagrees with reference (any disagreement)
        disagree_idx = np.where((rel_err > eps_machine * 0.1) & np.isfinite(vals))[0]
        if len(disagree_idx) > 0:
            print(
                f"  smallest |x| with finite-but-nonzero rel_err > 0.1·eps_machine: "
                f"{ts[disagree_idx[0]]:.2e}"
            )
            print(f"  largest such |x|: {ts[disagree_idx[-1]]:.2e}")


def show_auto_eps():
    """What does auto_eps pick for sinc, given there's no real breakdown?"""
    from taylor_remainder import TaylorAtPoint, auto_eps

    import pytensor.tensor as pt

    print("\n=== auto_eps for sinc(x) at order=10 ===")
    for dtype_pt, dtype_np in [("float64", np.float64), ("float32", np.float32)]:
        x = pt.scalar("x", dtype=dtype_pt)
        cache = TaylorAtPoint(pt.sin(x), x, 0.0)
        eps = auto_eps(cache, n=1, order=10)
        print(f"  dtype={dtype_pt}  auto_eps={eps:.3g}")
    print("  (whereas naive only fails at x=0; auto_eps is conservative here)")


if __name__ == "__main__":
    main()
    show_auto_eps()
