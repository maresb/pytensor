"""Compare three approaches to k-th derivative of taylor_remainder at x=a.

Option 0 (baseline): switch + iterated pt.grad + canonicalize
Option 1: polynomial branch only
Option 2: closed-form recurrence
              deriv(f, n, k) = Σ_{j=0..k} c_j(k, n) · tr(f^(k-j), n+j)
          with coefficients c_j(k, n) from a small recurrence.

All three share a TaylorAtPoint cache, so f^(m)(a) is computed once.
"""

import math
import time

from taylor_remainder import (
    TaylorAtPoint,
    auto_eps,
    taylor_remainder,
    taylor_remainder_poly,
)

import pytensor
import pytensor.tensor as pt
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.graph.traversal import ancestors


pytensor.config.on_opt_error = "ignore"


def nodes(v):
    return sum(1 for _ in ancestors([v]))


class RecurrenceBuilder:
    """k-th derivative of taylor_remainder(f, x, a, n) as
    Σ_{j=0..k} c_j(k, n) · taylor_remainder(f^(k-j), n+j),
    sharing a TaylorAtPoint cache for all f^(m)(a) values.
    """

    def __init__(self, f, x, a, n, order=10, eps=None, cache=None):
        self.cache = cache if cache is not None else TaylorAtPoint(f, x, a)
        self.x = x
        self.a = a
        self.n = n
        self.order = order
        self.eps = eps if eps is not None else auto_eps(self.cache, n, order)
        self._coef_memo = {}
        self._tr_memo = {}

    def coefs(self, k, n):
        if (k, n) in self._coef_memo:
            return self._coef_memo[(k, n)]
        if k == 0:
            res = [1]
        else:
            prev = self.coefs(k - 1, n)
            prev_n1 = self.coefs(k - 1, n + 1)
            res = [prev[0]]
            for j in range(1, k + 1):
                left = prev[j] if j < len(prev) else 0
                right = prev_n1[j - 1] if (j - 1) < len(prev_n1) else 0
                res.append(left - n * right)
        self._coef_memo[(k, n)] = res
        return res

    def tr(self, m, n_eff):
        """taylor_remainder(f^(m), x, a, n_eff), built from cached f^(m)(a) values."""
        key = (m, n_eff)
        if key in self._tr_memo:
            return self._tr_memo[key]
        coeffs = self.cache.coeffs_of_deriv(m, n_eff + self.order)
        t = self.x - self.a
        poly = coeffs[n_eff]
        for k in range(1, self.order):
            poly = poly + coeffs[n_eff + k] * t**k
        if n_eff == 0:
            closed = self.cache.deriv(m)
        else:
            P = coeffs[0]
            for k in range(1, n_eff):
                P = P + coeffs[k] * t**k
            closed = (self.cache.deriv(m) - P) / t**n_eff
        result = pt.switch(pt.abs(t) < self.eps, poly, closed)
        self._tr_memo[key] = result
        return result

    def build(self, k):
        cs = self.coefs(k, self.n)
        result = None
        for j in range(k + 1):
            m = k - j
            term = cs[j] * self.tr(m, self.n + j)
            result = term if result is None else result + term
        return result


def main():
    x = pt.dscalar("x")
    f = pt.log1p(x)
    K = 11

    def ref_at_0(k):
        return (-1) ** k * math.factorial(k) / (k + 1)

    def matches(val, ref):
        return abs(val - ref) <= 1e-9 * max(1.0, abs(ref))

    print(f"=== {K + 1} derivatives of psi_1[log1p](x) at x=0 ===\n")

    # Shared cache used by all three options. (Per-option fresh cache too -- we
    # measure each in isolation, but sharing across them would be even faster
    # for a real pipeline.)

    # ----- Option 0 -----
    K0 = min(K, 7)
    print(
        f"Option 0: switch + iterated pt.grad  (capped at k={K0} -- numba 1000-tuple limit)"
    )
    t_total = 0
    cache = TaylorAtPoint(f, x, 0.0)
    cur = taylor_remainder(f, x, 0.0, 1, order=10, cache=cache)
    cur = rewrite_graph(cur, include=("canonicalize",))
    for k in range(K0 + 1):
        t0 = time.time()
        fn = pytensor.function([x], cur)
        val = float(fn(0.0))
        elapsed = time.time() - t0
        t_total += elapsed
        ref = ref_at_0(k)
        ok = "ok" if matches(val, ref) else "FAIL"
        print(
            f"  k={k} {ok} val={val:>14.8g} step={elapsed * 1000:>6.0f}ms nodes={nodes(cur)}"
        )
        if k < K0:
            cur = pt.grad(cur, x)
            cur = rewrite_graph(cur, include=("canonicalize",))
    print(f"  total: {t_total * 1000:.0f}ms\n")

    # ----- Option 1 -----
    print(f"Option 1: polynomial only  (order={K + 2})")
    t_total = 0
    cache = TaylorAtPoint(f, x, 0.0)
    cur = taylor_remainder_poly(f, x, 0.0, 1, order=K + 2, cache=cache)
    cur = rewrite_graph(cur, include=("canonicalize",))
    for k in range(K + 1):
        t0 = time.time()
        fn = pytensor.function([x], cur)
        val = float(fn(0.0))
        elapsed = time.time() - t0
        t_total += elapsed
        ref = ref_at_0(k)
        ok = "ok" if matches(val, ref) else "FAIL"
        print(
            f"  k={k} {ok} val={val:>14.8g} step={elapsed * 1000:>6.0f}ms nodes={nodes(cur)}"
        )
        if k < K:
            cur = pt.grad(cur, x)
            cur = rewrite_graph(cur, include=("canonicalize",))
    print(f"  total: {t_total * 1000:.0f}ms\n")

    # ----- Option 2 -----
    print("Option 2: recurrence (TaylorAtPoint memoized)")
    t_total = 0
    cache = TaylorAtPoint(f, x, 0.0)
    builder = RecurrenceBuilder(f, x, 0.0, 1, order=10, cache=cache)
    for k in range(K + 1):
        t0 = time.time()
        cur = builder.build(k)
        cur = rewrite_graph(cur, include=("canonicalize",))
        fn = pytensor.function([x], cur)
        val = float(fn(0.0))
        elapsed = time.time() - t0
        t_total += elapsed
        ref = ref_at_0(k)
        ok = "ok" if matches(val, ref) else "FAIL"
        print(
            f"  k={k} {ok} val={val:>14.8g} step={elapsed * 1000:>6.0f}ms nodes={nodes(cur)}"
        )
    print(f"  total: {t_total * 1000:.0f}ms")


if __name__ == "__main__":
    main()
