"""Thorough check of naive K·sin(x)/x for various scales K.

Two evaluation orders:
  order_A:  (K · sin(x)) / x        -- can over/underflow in the K·sin(x) step
  order_B:  K · (sin(x) / x)        -- avoids over/underflow
  taylor_remainder: our switch-based stable form

Sweeps every binade x {zero, all-ones, alt 0101, alt 1010} mantissa patterns,
in float64.
"""

import math
import struct

import mpmath as mp
import numpy as np
from taylor_remainder import taylor_remainder

import pytensor
import pytensor.tensor as pt


mp.mp.dps = 50

F64_PATTERNS = [0x0, 0xFFFFFFFFFFFFF, 0x5555555555555, 0xAAAAAAAAAAAAA]


def f64_from_bits(biased_exp, mantissa):
    bits = ((biased_exp & 0x7FF) << 52) | (mantissa & ((1 << 52) - 1))
    return struct.unpack("!d", struct.pack("!Q", bits))[0]


def points_f64():
    pts = []
    for biased_exp in range(0, 0x7FF):
        for mantissa in F64_PATTERNS:
            v = f64_from_bits(biased_exp, mantissa)
            if v == 0 or not math.isfinite(v):
                continue
            pts.append(v)
    return pts


def naive_order_A(K, x):
    """(K · sin(x)) / x ."""
    return (K * np.sin(x)) / x


def naive_order_B(K, x):
    """K · (sin(x) / x) ."""
    return K * (np.sin(x) / x)


def reference(K, x):
    """High-precision K · sin(x) / x  via mpmath."""
    return float(mp.mpf(K) * mp.sin(mp.mpf(x)) / mp.mpf(x))


def taylor_remainder_fn(K):
    x = pt.dscalar("x")
    f = K * pt.sin(x)
    y = taylor_remainder(f, x, 0.0, 1, order=10)
    return pytensor.function([x], y)


def check(K, pts, tr_fn, eps_machine):
    fails = {
        "A_under": 0,
        "A_over": 0,
        "A_other": 0,
        "B_other": 0,
        "TR": 0,
        "A_total": 0,
        "B_total": 0,
        "TR_total": 0,
    }
    examples = {"A": [], "B": [], "TR": []}
    tol = 100 * eps_machine

    for x in pts:
        ref = reference(K, x)
        with np.errstate(all="ignore"):
            a = float(naive_order_A(K, x))
            b = float(naive_order_B(K, x))
            tr = float(tr_fn(x))

        for label, val, total_key, examples_key in [
            ("A", a, "A_total", "A"),
            ("B", b, "B_total", "B"),
            ("TR", tr, "TR_total", "TR"),
        ]:
            # true relative error (or absolute if ref=0)
            if ref == 0.0:
                bad = val != 0.0
                rel = abs(val)
            else:
                if not math.isfinite(val):
                    bad = True
                    rel = float("inf")
                else:
                    rel = abs(val - ref) / abs(ref)
                    bad = rel > tol

            if bad:
                fails[total_key] += 1
                if label == "A":
                    if val == 0.0 and ref != 0.0:
                        fails["A_under"] += 1
                    elif math.isinf(val):
                        fails["A_over"] += 1
                    else:
                        fails["A_other"] += 1
                if len(examples[examples_key]) < 3:
                    examples[examples_key].append((x, val, ref))

    return fails, examples


def main():
    pts = points_f64()
    eps_machine = float(np.finfo(np.float64).eps)
    print(f"swept {len(pts)} float64 points (every binade x 4 mantissa patterns)\n")

    # K values: full range from tiny subnormal up to near-max
    Ks = [
        1e-308,
        1e-200,
        1e-100,
        1e-50,
        1e-10,
        1e-3,
        1.0,
        1e3,
        1e10,
        1e50,
        1e100,
        1e200,
        1e300,
    ]

    print(
        f"{'K':>10} {'A:(K·sin)/x':>14} {'B:K·(sin/x)':>14} {'taylor_rem':>14}  examples"
    )
    print("-" * 90)
    for K in Ks:
        tr_fn = taylor_remainder_fn(K)
        fails, examples = check(K, pts, tr_fn, eps_machine)
        a_breakdown = (
            f"{fails['A_total']}"
            + (f" (uf={fails['A_under']})" if fails["A_under"] else "")
            + (f" (of={fails['A_over']})" if fails["A_over"] else "")
        )
        print(
            f"{K:>10.0e} {a_breakdown:>14} {fails['B_total']:>14} "
            f"{fails['TR_total']:>14}"
        )
        for label, exs in examples.items():
            if not exs:
                continue
            for x, val, ref in exs[:1]:
                print(f"             [{label}] x={x:.3e}  got={val:.3e}  ref={ref:.3e}")


if __name__ == "__main__":
    main()
