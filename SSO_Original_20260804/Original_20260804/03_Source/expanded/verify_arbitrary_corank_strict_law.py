#!/usr/bin/env python3
"""Assertion-based tests for the arbitrary-corank strict displacement law.

Requires only NumPy and SciPy. Every test has lambda_star = 0 and constructs
Theta in singular coordinates of A_star. The prediction is reconstructed
from an SVD of the resulting matrices, not from the construction parameters.
"""

import numpy as np
from numpy.linalg import qr, svd
from scipy.optimize import brentq


RTOL = 1.0e-13
ATOL = 2.0e-12
DELTAS = np.array([1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6])


def orthogonal(rng, size):
    """Deterministic orthogonal matrix, with QR signs canonicalized."""
    q, r = qr(rng.standard_normal((size, size)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def make_case(seed, m, n, positive_sv, c_sv, a_target, name):
    """Construct A_star and Theta in fixed singular coordinates."""
    rng = np.random.default_rng(seed)
    r = len(positive_sv)
    qdim = min(m, n)
    left_nullity, right_nullity = m - r, n - r
    assert r < qdim
    assert len(c_sv) <= min(left_nullity, right_nullity)

    u = orthogonal(rng, m)
    v = orthogonal(rng, n)

    d = np.zeros((m, n))
    d[np.arange(r), np.arange(r)] = positive_sv
    a_star = u @ d @ v.T

    # The kernel compression has precisely the prescribed positive singular
    # values but is non-diagonal in its own left/right null coordinates.
    pc = orthogonal(rng, left_nullity)
    qc = orthogonal(rng, right_nullity)
    c_diag = np.zeros((left_nullity, right_nullity))
    rc = len(c_sv)
    c_diag[np.arange(rc), np.arange(rc)] = c_sv
    c_block = pc @ c_diag @ qc.T

    t11 = rng.standard_normal((r, r))
    t11 += ((a_target - np.trace(t11)) / r) * np.eye(r)
    t12 = 0.25 * rng.standard_normal((r, right_nullity))
    t21 = 0.25 * rng.standard_normal((left_nullity, r))
    theta_coordinates = np.block([[t11, t12], [t21, c_block]])
    theta = u @ theta_coordinates @ v.T

    return {
        "name": name,
        "A": a_star,
        "Theta": theta,
        "rank": r,
        "expected_rank_C": rc,
        "repeated_positive": len(set(np.asarray(positive_sv))) < r,
    }


def z_delta(a, delta):
    u, singular_values, vt = svd(a, full_matrices=False)
    coefficients = singular_values / np.sqrt(singular_values**2 + delta**2)
    return (u * coefficients) @ vt


def reconstruct_data(case):
    """Reconstruct a, C, beta from a full SVD, as in the theorem."""
    a_star, theta, r = case["A"], case["Theta"], case["rank"]
    u, singular_values, vt = svd(a_star, full_matrices=True)
    assert singular_values[r - 1] > 1.0e-8
    assert r == len(singular_values) or singular_values[r] < 1.0e-10

    compact_polar = u[:, :r] @ vt[:r, :]
    u0 = u[:, r:]
    v0 = vt[r:, :].T
    a = float(np.sum(theta * compact_polar))
    c = u0.T @ theta @ v0
    c_singular_values = svd(c, compute_uv=False)
    positive = c_singular_values[c_singular_values > 1.0e-10]
    beta = float(np.sum(positive))

    assert len(positive) == case["expected_rank_C"]
    assert abs(a) < beta - 1.0e-8
    return a, c, positive, beta


def eta_prediction(a, c_singular_values):
    if abs(a) <= 1.0e-14:
        return 0.0

    def equation(eta):
        terms = eta * c_singular_values**2
        terms /= np.sqrt(1.0 + (eta * c_singular_values) ** 2)
        return float(np.sum(terms) - abs(a))

    upper = 1.0
    while equation(upper) < 0.0:
        upper *= 2.0
        assert upper < 1.0e12
    eta = brentq(equation, 0.0, upper, xtol=ATOL, rtol=RTOL)
    return -np.sign(a) * eta


def smooth_root(a_star, theta, delta, predicted_ratio):
    def h(lam):
        return float(np.sum(theta * z_delta(a_star + lam * theta, delta)))

    width = max(1.0e-6, 8.0 * (1.0 + abs(predicted_ratio)) * delta)
    left, right = -width, width
    f_left, f_right = h(left), h(right)
    for _ in range(80):
        if f_left <= 0.0 <= f_right:
            break
        if f_left > 0.0:
            left *= 2.0
            f_left = h(left)
        if f_right < 0.0:
            right *= 2.0
            f_right = h(right)
    else:
        raise AssertionError("failed to bracket the unique smooth root")

    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right
    return brentq(h, left, right, xtol=ATOL, rtol=RTOL)


def verify(case):
    a, c, c_singular_values, beta = reconstruct_data(case)
    prediction = eta_prediction(a, c_singular_values)
    ratios = np.array([
        smooth_root(case["A"], case["Theta"], delta, prediction) / delta
        for delta in DELTAS
    ])
    errors = np.abs(ratios - prediction)

    # The first three reductions should show asymptotic improvement. The
    # final tolerance leaves ample room above double-precision SVD/root noise.
    assert errors[-1] < errors[0]
    assert errors[-1] <= 2.0e-5 * (1.0 + abs(prediction)), (
        case["name"], prediction, ratios, errors
    )

    m, n = case["A"].shape
    qdim = min(m, n)
    rank_c = len(c_singular_values)
    print(
        f"PASS {case['name']:<32} shape={m}x{n} "
        f"corank={qdim-case['rank']} rank(C)={rank_c} "
        f"a={a:+.8f} beta={beta:.8f} prediction={prediction:+.10f} "
        f"ratio(delta=1e-6)={ratios[-1]:+.10f}"
    )
    return prediction, ratios


def main():
    cases = [
        make_case(
            seed=101,
            m=5,
            n=3,
            positive_sv=[1.7],
            c_sv=[1.2, 0.7],
            a_target=0.45,
            name="tall unequal nullities",
        ),
        make_case(
            seed=202,
            m=3,
            n=5,
            positive_sv=[1.4],
            c_sv=[1.1, 0.6],
            a_target=-0.40,
            name="wide unequal nullities",
        ),
        make_case(
            seed=303,
            m=6,
            n=4,
            positive_sv=[1.6],
            c_sv=[1.3],
            a_target=0.35,
            name="corank-3 rank-deficient C",
        ),
        make_case(
            seed=404,
            m=6,
            n=6,
            positive_sv=[1.0, 1.0],
            c_sv=[1.1, 0.8, 0.5],
            a_target=0.60,
            name="repeated positive cluster",
        ),
        make_case(
            seed=505,
            m=4,
            n=6,
            positive_sv=[1.5, 0.9],
            c_sv=[1.25],
            a_target=0.0,
            name="zero-a vanishing coefficient",
        ),
    ]

    results = [verify(case) for case in cases]

    # Structural coverage assertions, independent of human-readable names.
    assert any(case["A"].shape[0] > case["A"].shape[1] for case in cases)
    assert any(case["A"].shape[0] < case["A"].shape[1] for case in cases)
    assert any(
        min(case["A"].shape) - case["rank"] > 1
        and case["expected_rank_C"] < min(case["A"].shape) - case["rank"]
        for case in cases
    )
    assert any(case["repeated_positive"] for case in cases)
    assert all(np.all(np.isfinite(ratios)) for _, ratios in results)
    print("All arbitrary-corank strict-law assertions passed.")


if __name__ == "__main__":
    main()
