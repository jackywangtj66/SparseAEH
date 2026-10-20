"""Profiled covariance fitting for the block-conditional Gaussian mixture."""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar


def _conditioning_projections(deviations, kernel):
    """Cache delta-independent projections for one fixed set of deviations."""
    deviations = np.asarray(deviations, dtype=float)
    if deviations.ndim != 2 or deviations.shape[0] != kernel.N:
        raise ValueError("deviations must have shape (locations, features)")
    return tuple(
        kernel.ds_eig[i][1].T @ deviations[locations] if locations else None
        for i, locations in enumerate(kernel.ds_loc)
    )


def block_statistics(deviations, kernel, delta, *, _projections=None):
    """Return the block log determinant and per-feature quadratic forms."""
    deviations = np.asarray(deviations, dtype=float)
    if deviations.ndim != 2 or deviations.shape[0] != kernel.N:
        raise ValueError("deviations must have shape (locations, features)")
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")

    log_determinant = 0.0
    quadratic = np.zeros(deviations.shape[1], dtype=float)
    for i, locations in enumerate(kernel.ss_loc):
        residual = deviations[locations]
        covariance = np.array(kernel.full_cov[i][i], dtype=float, copy=True)
        covariance.flat[::len(locations) + 1] += delta
        if kernel.ds_loc[i]:
            eigenvalues, eigenvectors = kernel.ds_eig[i]
            shifted = eigenvalues + delta
            if np.any(shifted <= 0):
                raise np.linalg.LinAlgError("conditioning covariance is not positive definite")
            cross_eigenvectors = kernel.A[i]
            inverse_cross = cross_eigenvectors / shifted
            covariance -= inverse_cross @ cross_eigenvectors.T
            if _projections is not None:
                residual = residual - inverse_cross @ _projections[i]
            elif deviations.shape[1] > len(locations):
                # Form the small block-by-conditioning predictor first.
                residual = residual - (inverse_cross @ eigenvectors.T) @ deviations[kernel.ds_loc[i]]
            else:
                residual = residual - inverse_cross @ (
                    eigenvectors.T @ deviations[kernel.ds_loc[i]]
                )
        covariance = (covariance + covariance.T) * 0.5
        factor = np.linalg.cholesky(covariance)
        whitened = solve_triangular(factor, residual, lower=True, check_finite=False)
        log_determinant += 2 * np.log(np.diag(factor)).sum()
        quadratic += np.square(whitened).sum(axis=0)
    return float(log_determinant), quadratic


def _profiled_from_statistics(log_determinant, quadratic, weights, total_weight,
                              n_locations, scale_floor):
    energy = float(weights @ quadratic)
    scale = max(scale_floor, energy / (n_locations * total_weight))
    objective = (total_weight * n_locations * np.log(scale)
                 + total_weight * log_determinant + energy / scale)
    return float(objective), float(scale)


def profiled_objective(deviations, weights, kernel, delta, scale_floor, *,
                       _projections=None):
    """Minimize the negative twice-Q covariance terms over the scale at delta."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != deviations.shape[1]:
        raise ValueError("weights must have one entry per feature")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")
    if not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be positive and finite")
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("component weight must be positive")
    log_determinant, quadratic = block_statistics(
        deviations, kernel, delta, _projections=_projections
    )
    return _profiled_from_statistics(
        log_determinant, quadratic, weights, total_weight, kernel.N, scale_floor
    )


def line_search_covariance(deviations, weights, kernel, current_scale, current_delta,
                           delta_bounds=(0.0, 10.0), scale_floor=1e-8,
                           grid_size=12, refinements=3, xatol=1e-4,
                           acceptance_tolerance=1e-10, return_statistics=False):
    """Search the nugget and accept only a nondecreasing fixed-responsibility Q.

    The grid includes the current value and both bounds. Positive values are
    logarithmically spaced; local minima are refined with bounded Brent search.
    No unimodality assumption is made. Conditioning projections are cached only
    for this call, using G * sum(len(C_m)) floating-point entries.
    If requested, also return (log determinant, per-feature quadratic forms)
    for the returned parameters. Only baseline and best statistics are retained.
    """
    lower, upper = map(float, delta_bounds)
    if not (0 <= lower < upper and lower <= current_delta <= upper):
        raise ValueError("delta_bounds must contain the current nonnegative delta")
    if not np.isfinite(current_scale) or current_scale < scale_floor:
        raise ValueError("current_scale must be finite and at least scale_floor")
    if grid_size < 3 or refinements < 0 or xatol <= 0 or acceptance_tolerance < 0:
        raise ValueError("invalid line-search settings")

    weights = np.asarray(weights, dtype=float)
    deviations = np.asarray(deviations, dtype=float)
    if weights.ndim != 1 or deviations.ndim != 2 or weights.shape[0] != deviations.shape[1]:
        raise ValueError("weights must have one entry per feature")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")
    if not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be positive and finite")
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("component weight must be positive")
    # Deviations are fixed throughout this search, but change between M-steps.
    projections = _conditioning_projections(deviations, kernel)
    old_logdet, old_quadratic = block_statistics(
        deviations, kernel, current_delta, _projections=projections
    )
    old_energy = float(weights @ old_quadratic)
    old_objective = (total_weight * kernel.N * np.log(current_scale)
                     + total_weight * old_logdet + old_energy / current_scale)

    evaluations = {}
    best_statistics = None
    best_score = np.inf

    def evaluate(value):
        nonlocal best_statistics, best_score
        value = float(value)
        if value not in evaluations:
            try:
                statistics = ((old_logdet, old_quadratic) if value == current_delta
                              else block_statistics(
                                  deviations, kernel, value, _projections=projections
                              ))
                evaluations[value] = _profiled_from_statistics(
                    *statistics, weights, total_weight, kernel.N, scale_floor
                )
                if evaluations[value][0] < best_score:
                    best_score = evaluations[value][0]
                    best_statistics = statistics
            except np.linalg.LinAlgError:
                evaluations[value] = (np.inf, np.nan)
        return evaluations[value]

    positive_lower = lower if lower > 0 else min(1e-6, upper)
    grid = np.geomspace(positive_lower, upper, grid_size)
    candidates = np.unique(np.r_[lower, upper, current_delta, grid])
    scores = np.array([evaluate(value)[0] for value in candidates])

    intervals = []
    for i in range(1, len(candidates) - 1):
        if scores[i] <= scores[i - 1] and scores[i] <= scores[i + 1]:
            intervals.append((scores[i], candidates[i - 1], candidates[i + 1]))
    if not intervals and len(candidates) > 2:
        i = int(np.argmin(scores))
        if i == 0:
            intervals.append((scores[i], candidates[0], candidates[1]))
        elif i == len(candidates) - 1:
            intervals.append((scores[i], candidates[-2], candidates[-1]))
    for _, left, right in sorted(intervals)[:refinements]:
        result = minimize_scalar(lambda value: evaluate(value)[0],
                                 bounds=(left, right), method='bounded',
                                 options={'xatol': xatol})
        if np.isfinite(result.x):
            evaluate(result.x)

    best_delta = min(evaluations, key=lambda value: evaluations[value][0])
    best_objective, best_scale = evaluations[best_delta]
    accepted = (np.isfinite(best_objective)
                and best_objective <= old_objective
                + acceptance_tolerance * max(1.0, abs(old_objective)))
    result = {
        'sigma_sq': best_scale if accepted else float(current_scale),
        'delta': best_delta if accepted else float(current_delta),
        'accepted': bool(accepted),
        'objective_before': float(old_objective),
        'objective_after': float(best_objective if accepted else old_objective),
        'evaluations': len(evaluations),
        'at_boundary': bool(accepted and (best_delta == lower or best_delta == upper)),
    }
    if return_statistics:
        return result, best_statistics if accepted else (old_logdet, old_quadratic)
    return result
