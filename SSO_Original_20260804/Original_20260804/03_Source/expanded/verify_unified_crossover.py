#!/usr/bin/env python3
"""Assertion-based, dependency-free high-precision crossover checks.

The test family is

    A(lambda,c) = [[lambda, 1], [1, c]],  Theta = e_1 e_1^T,

with its singular minimizer at lambda_sing=1/c.  We parameterize it by

    Delta = (c^2-1)/(c^2+1),

so that beta=(1+Delta)/2 and
kappa=(1/2)(1-Delta^2)^(3/2).  Decimal arithmetic is used so the script
has no NumPy/SciPy/mpmath dependency.  A failed asymptotic check exits with
a nonzero status through an AssertionError.
"""

from decimal import Decimal, getcontext


getcontext().prec = 100
D = Decimal
ONE = D(1)
TWO = D(2)
HALF = ONE / TWO


def absd(x):
    return x.copy_abs()


def family_data(delta_gap):
    """Return c, beta, kappa, and the exact crossing from Delta."""
    assert D(0) <= delta_gap < D(1)
    c = ((ONE + delta_gap) / (ONE - delta_gap)).sqrt()
    beta = (ONE + delta_gap) / TWO
    kappa = HALF * ((ONE - delta_gap * delta_gap).sqrt() ** 3)
    crossing = ONE / c
    return c, beta, kappa, crossing


def root_sign(lam, c, delta):
    """Sign-equivalent numerator of <Theta,Z_delta(A(lambda,c))>."""
    s00 = lam * lam + ONE + delta * delta
    s01 = lam + c
    s11 = ONE + c * c + delta * delta
    det_s = s00 * s11 - s01 * s01
    assert det_s > 0
    root_det = det_s.sqrt()
    # Z_00 has the same sign as this numerator.
    return lam * (s11 + root_det) - s01


def smoothed_root(c, delta, crossing):
    """Bisection for the unique smoothed root on the contact side."""
    hi = crossing
    f_hi = root_sign(hi, c, delta)
    assert f_hi > 0
    width = ONE
    lo = crossing - width
    f_lo = root_sign(lo, c, delta)
    while f_lo >= 0:
        width *= TWO
        lo = crossing - width
        f_lo = root_sign(lo, c, delta)
    for _ in range(420):
        mid = (lo + hi) / TWO
        if root_sign(mid, c, delta) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / TWO


def positive_cubic_root(kappa, gap, rhs):
    """Positive root of kappa*x^3+gap*x^2=rhs by bisection."""
    lo = D(0)
    hi = ONE
    while kappa * hi**3 + gap * hi**2 <= rhs:
        hi *= TWO
    for _ in range(420):
        mid = (lo + hi) / TWO
        if kappa * mid**3 + gap * mid**2 < rhs:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / TWO


def predicted_y(zeta):
    # At the boundary beta_0=kappa_0=1/2:
    # (1/2)y^3+zeta*y^2=1.
    return positive_cubic_root(HALF, zeta, ONE)


def invsqrt_2x2(s00, s01, s11):
    """Principal inverse square root of a 2x2 SPD matrix."""
    root_det = (s00 * s11 - s01 * s01).sqrt()
    scale = (s00 + s11 + TWO * root_det).sqrt()
    c00 = s00 + root_det
    c11 = s11 + root_det
    det_c = c00 * c11 - s01 * s01
    return (
        scale * c11 / det_c,
        -scale * s01 / det_c,
        scale * c00 / det_c,
    )


def phi_delta(lam, c, delta):
    """Z_delta([[lambda,1],[1,c]]) as a four-tuple."""
    s00 = lam * lam + ONE + delta * delta
    s01 = lam + c
    s11 = ONE + c * c + delta * delta
    m00, m01, m11 = invsqrt_2x2(s00, s01, s11)
    return (
        lam * m00 + m01,
        lam * m01 + m11,
        m00 + c * m01,
        m01 + c * m11,
    )


def exact_center(c):
    """Unique exact certificate for the square corank-one family."""
    den = ONE + c * c
    p = (ONE / den, c / den, c / den, c * c / den)
    d = (c * c / den, -c / den, -c / den, ONE / den)
    k = -ONE / (c * c)
    return tuple(p_i + k * d_i for p_i, d_i in zip(p, d))


def frob_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)).sqrt()


def solve_path(gap, delta):
    c, beta, kappa, crossing = family_data(gap)
    lam = smoothed_root(c, delta, crossing)
    x = crossing - lam
    xhat = positive_cubic_root(kappa, gap, delta * delta / (TWO * beta))
    direction_error = frob_distance(phi_delta(lam, c, delta), exact_center(c))
    return x, xhat, beta, kappa, direction_error


def decreasing(values, label):
    for left, right in zip(values, values[1:]):
        assert right < left, f"{label} did not decrease: {values}"


def main():
    print("Critical window: Delta=zeta*delta^(2/3)")
    for zeta in map(D, (0, 1, 3, 10)):
        target = predicted_y(zeta)
        errors = []
        xhat_errors = []
        rows = []
        for k in (2, 3, 4, 5):
            q = D(10) ** (-k)       # delta^(1/3)
            delta = q**3
            gap = zeta * q**2
            x, xhat, _, _, _ = solve_path(gap, delta)
            ratio = x / q**2
            rows.append(ratio)
            errors.append(absd(ratio - target))
            xhat_errors.append(absd(x / xhat - ONE))
        decreasing(errors, f"critical zeta={zeta}")
        decreasing(xhat_errors, f"unified critical zeta={zeta}")
        assert errors[-1] < D("8e-5")
        assert xhat_errors[-1] < D("8e-5")
        print(f"  zeta={zeta}: y[-1]={rows[-1]:.12E}, target={target:.12E}, "
              f"x/xhat[-1]={ONE + (x / xhat - ONE):.12E}")

    print("Boundary path: Delta=0")
    target = TWO ** (ONE / D(3))
    errors = []
    direction_errors = []
    mnorm = (D(3) / TWO).sqrt()
    for k in (2, 3, 4, 5):
        q = D(10) ** (-k)
        delta = q**3
        x, xhat, _, _, derr = solve_path(D(0), delta)
        errors.append(absd(x / q**2 - target))
        direction_errors.append(absd(derr / x - mnorm))
    decreasing(errors, "boundary displacement")
    decreasing(direction_errors, "boundary direction")
    assert errors[-1] < D("2e-5")
    assert direction_errors[-1] < D("2e-5")
    print(f"  x/delta^(2/3)={x/q**2:.12E}, target={target:.12E}")
    print(f"  ||Phi_delta-Phi*||/x={derr/x:.12E}, target=sqrt(3/2)={mnorm:.12E}")

    print("Strict-side path: Delta=delta^(1/3)")
    errors = []
    xhat_errors = []
    direction_errors = []
    values = []
    for k in (2, 3, 4, 5):
        q = D(10) ** (-k)
        delta = q**3
        gap = q
        x, xhat, _, _, derr = solve_path(gap, delta)
        normalized = x * gap.sqrt() / delta
        values.append(normalized)
        errors.append(absd(normalized - ONE))
        xhat_errors.append(absd(x / xhat - ONE))
        direction_errors.append(absd(derr / x - mnorm))
    decreasing(errors, "strict-side displacement")
    decreasing(xhat_errors, "unified strict-side")
    decreasing(direction_errors, "strict-side direction")
    assert errors[-1] < D("3e-5")
    assert xhat_errors[-1] < D("3e-5")
    assert direction_errors[-1] < D("3e-5")
    print("  normalized values:", ", ".join(f"{v:.9f}" for v in values))
    print(f"  final x/xhat={x/xhat:.12E}")
    print(f"  final ||Phi_delta-Phi*||/x={derr/x:.12E}")

    # One deliberately oscillatory path exercises the unified statement.
    print("Oscillatory path: alternating boundary, critical, and strict windows")
    ratios = []
    for k in range(2, 11):
        q = D(10) ** (-k)
        delta = q**3
        mode = k % 3
        gap = D(0) if mode == 0 else (D(3) * q**2 if mode == 1 else q)
        x, xhat, _, _, _ = solve_path(gap, delta)
        ratios.append(x / xhat)
    tail_error = max(absd(v - ONE) for v in ratios[-3:])
    assert tail_error < D("3e-7")
    print("  last three x/xhat:", ", ".join(f"{v:.12E}" for v in ratios[-3:]))

    print("ALL UNIFIED-CROSSOVER ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
