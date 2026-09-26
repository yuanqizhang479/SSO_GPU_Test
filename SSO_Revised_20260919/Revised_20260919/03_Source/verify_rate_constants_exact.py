#!/usr/bin/env python3
"""Dependency-free high-precision checks of the D1--D3 constants.

The instances are the exact 2-by-2 families printed in the manuscript.
Decimal arithmetic and bisection make the test deterministic.  A failed
constant or failed monotone-convergence check raises AssertionError.
"""

# Scientific assertions must never disappear under python -O / PYTHONOPTIMIZE.
if not __debug__:
    raise RuntimeError("Verification requires assertions: rerun without -O / PYTHONOPTIMIZE.")

from decimal import Decimal, getcontext


getcontext().prec = 100
D = Decimal
ZERO, ONE, TWO = D(0), D(1), D(2)


def root_sign(x, r, s, c, delta):
    """Sign-equivalent numerator of [Z_delta([[x,r],[s,c]])]_(1,1)."""
    s00 = x * x + s * s + delta * delta
    s01 = x * r + s * c
    s11 = r * r + c * c + delta * delta
    det_s = s00 * s11 - s01 * s01
    assert det_s > 0
    root_det = det_s.sqrt()
    return x * (s11 + root_det) - r * s01


def smooth_root(r, s, c, delta, lo=D(-2), hi=D(2)):
    f_lo = root_sign(lo, r, s, c, delta)
    f_hi = root_sign(hi, r, s, c, delta)
    assert f_lo < 0 < f_hi
    for _ in range(430):
        mid = (lo + hi) / TWO
        if root_sign(mid, r, s, c, delta) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / TWO


def strictly_decreasing(values, label):
    for left, right in zip(values, values[1:]):
        assert right < left, f"{label} did not converge monotonically: {values}"


def main():
    print("D1 regular law")
    target_d1 = -D(16) / D(9)
    errors = []
    for delta in map(D, ("1e-3", "1e-4", "1e-5")):
        root = smooth_root(ONE, ONE, ONE / TWO, delta)
        ratio = (root - ONE / TWO) / (delta * delta)
        errors.append(abs(ratio - target_d1))
        print(f"  delta={delta:.0E}  ratio={ratio:.12f}")
    strictly_decreasing(errors, "D1")
    assert errors[-1] < D("2e-8")

    print("D2 strict-kink law")
    target_d2 = -D(5) / (D(4) * D(15).sqrt())
    errors = []
    for delta in map(D, ("1e-5", "1e-6", "1e-7", "1e-8")):
        root = smooth_root(ONE / TWO, ONE / TWO, ONE, delta)
        ratio = (root - ONE / D(4)) / delta
        errors.append(abs(ratio - target_d2))
        print(f"  delta={delta:.0E}  ratio={ratio:.12f}")
    strictly_decreasing(errors, "D2")
    assert errors[-1] < D("3e-9")

    print("D3 boundary two-thirds law")
    target_d3 = TWO ** (ONE / D(3))
    errors = []
    for k in (2, 3, 4, 5):
        q = D(10) ** (-k)  # delta^(1/3)
        delta = q**3
        root = smooth_root(ONE, ONE, ONE, delta)
        ratio = (ONE - root) / (q * q)
        errors.append(abs(ratio - target_d3))
        print(f"  delta={delta:.0E}  ratio={ratio:.12f}")
    strictly_decreasing(errors, "D3")
    assert errors[-1] < D("2e-9")

    print("ALL D1--D3 HIGH-PRECISION ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
