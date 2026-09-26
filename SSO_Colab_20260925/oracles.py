"""Numerical reference oracles for the manuscript's explicitly defined path.

Z_delta(A)=U diag(s/hypot(s,delta)) V^T. Delta is FIXED along each
lambda search; no lambda-dependent normalization is hidden in this mapping.
Float64 SVD references are not interval certificates or a fast optimizer.
"""
import math
import torch


@torch.no_grad()
def smooth_polar(a, delta):
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError('delta must be finite and positive')
    a = a.double()
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    return (u * (s / torch.hypot(s, s.new_tensor(delta)))) @ vh


@torch.no_grad()
def solve_ph(g, theta, delta, initial=0., residual_tol=2e-12,
             max_iter=100, max_expand=60):
    g, theta = g.double(), theta.double()
    if g.ndim != 2 or g.shape != theta.shape:
        raise ValueError('g and theta must be equally shaped matrices')
    if not torch.isfinite(g).all() or not torch.isfinite(theta).all():
        raise ValueError('nonfinite matrix input')
    tn = float(theta.norm())
    if tn == 0:
        raise ValueError('theta must be nonzero')
    calls = 0

    def evaluate(lam):
        nonlocal calls
        z = smooth_polar(g + lam * theta, delta)
        h = float((z * theta).sum())
        calls += 1
        if not math.isfinite(h):
            raise FloatingPointError('nonfinite scalar derivative')
        return h, z

    h, z = evaluate(initial)
    best = (abs(h), initial, z)
    if abs(h) <= residual_tol:
        return initial, z, dict(status='converged', residual=abs(h), calls=calls,
                               bracket_width=0., delta=delta)
    width = max(float(g.norm()) / tn, delta / tn, 1e-6)
    lo = hi = initial
    flo = fhi = h
    for _ in range(max_expand):
        if flo > 0:
            lo = initial - width
            flo, zl = evaluate(lo)
            if abs(flo) < best[0]: best = (abs(flo), lo, zl)
        if fhi < 0:
            hi = initial + width
            fhi, zh = evaluate(hi)
            if abs(fhi) < best[0]: best = (abs(fhi), hi, zh)
        if flo <= 0 <= fhi:
            break
        width *= 2
    else:
        return best[1], best[2], dict(status='bracket_failed', residual=best[0],
                                     calls=calls, bracket_width=hi-lo, delta=delta)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi: break
        fm, zm = evaluate(mid)
        if abs(fm) < best[0]: best = (abs(fm), mid, zm)
        if best[0] <= residual_tol: break
        if fm < 0: lo = mid
        else: hi = mid
    res, lam, z = best
    return lam, z, dict(status='converged' if res <= residual_tol else 'residual_limited',
                       residual=res, calls=calls, bracket_width=hi-lo, delta=delta)


@torch.no_grad()
def repair(z, theta):
    """For arbitrary nonzero theta: feasible in exact arithmetic, float64 diagnostic here."""
    z, theta = z.double(), theta.double()
    z = z / torch.linalg.matrix_norm(z, 2).clamp_min(1)
    c = (theta * z).sum() / theta.square().sum()
    return (z - c * theta) / (1 + c.abs() * torch.linalg.matrix_norm(theta, 2))


@torch.no_grad()
def assess(g, theta, z, lam):
    g, theta, z = g.double(), theta.double(), z.double()
    zr = repair(z, theta)
    a = g + lam * theta
    upper = float(torch.linalg.svdvals(a).sum())
    lower = float((g * zr).sum())
    # Do not clamp a negative gap: a numerical inconsistency must remain visible.
    return dict(tangency=float((theta*z).sum().abs()),
                spectral_norm=float(torch.linalg.matrix_norm(z, 2)),
                raw_objective=float((g*z).sum()), repaired_objective=lower,
                primal_upper=upper, repaired_gap=upper-lower,
                relative_repaired_gap=(upper-lower)/max(abs(upper), 1e-15),
                repair_frobenius_change=float((zr-z).norm()),
                repaired_tangency=float((theta*zr).sum().abs()),
                repaired_spectral_norm=float(torch.linalg.matrix_norm(zr, 2)))
