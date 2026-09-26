#!/usr/bin/env python3
"""Check the finite cancellation criterion using exact rational moments.

The algebraic part uses fractions, and the scalar pseudo-Huber roots use
100-digit mpmath arithmetic.  This is numerical verification of examples,
not interval arithmetic or a formal proof of the general corollary.
"""

from fractions import Fraction
import json

import mpmath as mp


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def moments(levels, weights, count):
    return [
        sum((Fraction(w) / Fraction(nu) ** (2 * k)
             for nu, w in zip(levels, weights)), Fraction(0))
        for k in range(count)
    ]


def smooth_tangency(lam, delta, diagonal, theta):
    return mp.fsum(
        w * (a + lam * w) / mp.sqrt((a + lam * w) ** 2 + delta ** 2)
        for a, w in zip(diagonal, theta)
    )


def smooth_root(delta, diagonal, theta):
    lo, hi = mp.mpf(-1), mp.mpf(1)
    fun = lambda lam: smooth_tangency(lam, delta, diagonal, theta)
    require(fun(lo) < 0 < fun(hi), "scalar root bracket failed")
    for _ in range(380):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        residual = fun(mid)
        if residual == 0:
            return mid
        if residual < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    mp.mp.dps = 100
    report = {"arithmetic": "exact Fraction moments; mpmath 100 digits"}

    # Repeated positive levels: each entire level has zero paired weight.
    diagonal = [mp.mpf(x) for x in [1, 1, 2, 2, 0]]
    theta = [mp.mpf(x) for x in [1, -1, 3, -3, 1]]
    residuals = []
    for delta in map(mp.mpf, ["0.000001", "0.1", "1", "100"]):
        residual = smooth_tangency(mp.mpf(0), delta, diagonal, theta)
        require(abs(residual) < mp.mpf("1e-90"), "levelwise cancellation failed")
        residuals.append(mp.nstr(residual, 12))
    report["repeated_levels_exact_root_residuals"] = residuals

    # Empty positive cluster (A*=0): stationarity is exact for every delta.
    for delta in map(mp.mpf, ["1e-6", "1", "100"]):
        require(smooth_tangency(mp.mpf(0), delta, [0, 0], [2, -3]) == 0,
                "empty positive cluster failed")

    cases = [
        {"name": "cubic_two_levels", "levels": [1, 2], "weights": [1, -1],
         "p": 1, "coefficient": Fraction(3, 8)},
        {"name": "quintic_three_levels", "levels": [1, 2, 3],
         "weights": [5, -32, 27], "p": 2, "coefficient": Fraction(-5, 4)},
    ]
    for case in cases:
        levels, weights, p = case["levels"], case["weights"], case["p"]
        chi = moments(levels, weights, len(levels))
        require(all(value == 0 for value in chi[:p]), "earlier moments must vanish")
        require(chi[p] != 0 and p <= len(levels) - 1, "first nonzero moment bound failed")
        diagonal = list(map(mp.mpf, levels + [0]))
        theta = list(map(mp.mpf, weights + [1]))
        ratios = []
        for delta in map(mp.mpf, ["1e-4", "1e-6"]):
            root = smooth_root(delta, diagonal, theta)
            ratios.append(root / delta ** (2 * p + 1))
        expected = mp.mpf(case["coefficient"].numerator) / case["coefficient"].denominator
        require(abs(ratios[-1] - expected) < mp.mpf("3e-7"),
                "asymptotic multiplier constant failed")
        report[case["name"]] = {
            "moments": [str(value) for value in chi],
            "predicted_order": 2 * p + 1,
            "predicted_coefficient": str(case["coefficient"]),
            "deltas": ["1e-4", "1e-6"],
            "computed_ratios": [mp.nstr(value, 35) for value in ratios],
        }

    print(json.dumps(report, indent=2))
    print("FINITE CANCELLATION CHECKS PASSED (numerical verification, not formal proof).")


if __name__ == "__main__":
    main()
