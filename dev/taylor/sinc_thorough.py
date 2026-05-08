"""Thorough check of naive sin(x)/x against mpmath, sweeping every binade
combined with several mantissa patterns: zero, all-ones, alternating 0101,
alternating 1010. In both float32 and float64.

This covers every "interesting" structural mantissa shape across the entire
representable range, including subnormals, smallest_normal boundary, and
largest finite.
"""

import math
import struct

import mpmath as mp
import numpy as np


mp.mp.dps = 50


# IEEE 754 layout
F64_MANTISSA_BITS = 52
F32_MANTISSA_BITS = 23
F64_BIAS = 1023
F32_BIAS = 127

F64_MANTISSA_PATTERNS = {
    "zeros": 0x0000000000000,
    "all_ones": 0xFFFFFFFFFFFFF,
    "alt_5555": 0x5555555555555,
    "alt_AAAA": 0xAAAAAAAAAAAAA,
}
F32_MANTISSA_PATTERNS = {
    "zeros": 0x000000,
    "all_ones": 0x7FFFFF,
    "alt_5555": 0x555555,
    "alt_AAAA": 0x2AAAAA,
}


def f64_from_bits(sign, biased_exp, mantissa):
    bits = (sign << 63) | ((biased_exp & 0x7FF) << 52) | (mantissa & ((1 << 52) - 1))
    return struct.unpack("!d", struct.pack("!Q", bits))[0]


def f32_from_bits(sign, biased_exp, mantissa):
    bits = (sign << 31) | ((biased_exp & 0xFF) << 23) | (mantissa & ((1 << 23) - 1))
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def thorough_test_points(dtype):
    """Yield every (label, value) for sweeping every biased exponent crossed
    with every interesting mantissa bit pattern, in `dtype`."""
    if dtype is np.float64:
        from_bits = f64_from_bits
        biased_exp_max = 0x7FF - 1  # 0x7FF reserved for inf/nan
        patterns = F64_MANTISSA_PATTERNS
    else:
        from_bits = f32_from_bits
        biased_exp_max = 0xFF - 1
        patterns = F32_MANTISSA_PATTERNS

    pts = []
    # biased_exp = 0 is subnormal (mantissa 0 -> 0; mantissa nonzero -> subnormals)
    # biased_exp >= 1 is normal
    for biased_exp in range(0, biased_exp_max + 1):
        for label, mantissa in patterns.items():
            v = from_bits(0, biased_exp, mantissa)
            if v == 0 or not math.isfinite(v):
                continue
            pts.append((f"e={biased_exp},m={label}", v))
    # A few specific irrational anchors
    for label, v in [
        ("1.0", 1.0),
        ("pi/2", math.pi / 2),
        ("pi", math.pi),
        ("e", math.e),
        ("100", 100.0),
        ("1e10", 1e10),
        ("1e20", 1e20),
    ]:
        if v <= float(np.finfo(dtype).max):
            pts.append((f"anchor:{label}", v))
    return pts


def reference(x):
    if x == 0:
        return 1.0
    return float(mp.sin(mp.mpf(x)) / mp.mpf(x))


def check(dtype):
    finfo = np.finfo(dtype)
    eps_machine = float(finfo.eps)
    pts = thorough_test_points(dtype)
    print(
        f"\n=== dtype={dtype.__name__}  "
        f"eps_machine={eps_machine:.2e}  "
        f"checked {len(pts)} points  ==="
    )

    fails = []
    for label, x_py in pts:
        with np.errstate(all="ignore"):
            naive = float(
                np.sin(np.asarray(x_py, dtype=dtype)) / np.asarray(x_py, dtype=dtype)
            )
        ref = reference(x_py)
        if not math.isfinite(naive):
            fails.append((label, x_py, naive, ref, "non-finite"))
            continue
        rel = abs(naive - ref) / max(1.0, abs(ref))
        if rel > 100 * eps_machine:
            fails.append((label, x_py, naive, ref, f"rel_err={rel:.2e}"))

    if not fails:
        print(f"  all {len(pts)} points pass within 100·eps_machine")
    else:
        print(f"  {len(fails)} failures:")
        for label, x_py, naive, ref, why in fails[:20]:
            print(
                f"    [{label}]  x={x_py:>12.4e}  naive={naive!r:>16}  "
                f"ref={ref:>14.6g}  {why}"
            )
        if len(fails) > 20:
            print(f"    ... ({len(fails) - 20} more)")


def main():
    check(np.float64)
    check(np.float32)


if __name__ == "__main__":
    main()
