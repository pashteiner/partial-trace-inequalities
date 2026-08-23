r"""Short quadratic programs for joint partial-trace bounds.

Pass a spectrum to :func:`solve_spectrum`. The dimension is inferred,
and every sufficient plateau pattern available for that dimension is solved.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, linprog, minimize


@dataclass(frozen=True)
class QPResult:
    """A structured majorizer returned by the quadratic program."""

    reduced: np.ndarray
    spectrum: np.ndarray
    multiplicities: tuple[int, ...]
    objective: float


def _spectrum(values: Sequence[float], d: int | None = None) -> tuple[np.ndarray, int]:
    """Validate, sort, and infer the equal local dimension."""

    lam = np.sort(np.asarray(values, dtype=float).reshape(-1))[::-1]
    inferred = isqrt(lam.size)
    d = inferred if d is None else d
    if not lam.size or not np.all(np.isfinite(lam)):
        raise ValueError("The spectrum must be a nonempty finite vector.")
    if d < 3 or d * d != lam.size:
        raise ValueError("The spectrum must have length d^2 for an integer d >= 3.")
    return lam, d


def plateau_patterns(d: int) -> dict[str, tuple[int, ...]]:
    """Return every sufficient plateau pattern for an equal d-by-d system.

    A tuple gives the multiplicities of the reduced variables. For example,
    ``(1, 6, 1, 1)`` expands ``(a,b,c,e)`` to ``(a,b,...,b,c,e)``.
    """

    if d < 3:
        raise ValueError("The implemented plateau conditions require d >= 3.")
    pairs = (
        [(before, d - before) for before in range(max(0, d - 3), 4)]
        if d <= 5
        else [(3, 3)]
    )
    n = d * d
    # The subscript records the one-based position where the plateau starts.
    mu_names = ["I", "II", "III", "IV"]
    return {
        f"mu_{mu_names[before]}": (1,) * before
        + (n - before - after,)
        + (1,) * after
        for before, after in pairs
    }


def _expand(reduced: np.ndarray, multiplicities: tuple[int, ...]) -> np.ndarray:
    return np.repeat(reduced, multiplicities)


def solve_majorizer(
    lambda_spectrum: Sequence[float],
    multiplicities: Sequence[int],
    *,
    nonnegative: bool = False,
    tolerance: float = 1e-8,
) -> QPResult:
    r"""Minimize ``||mu||_2^2`` subject to ``lambda \prec mu``.

    The reduced variables are decreasing and repeated according to
    ``multiplicities``. A linear program first finds a feasible point; SLSQP
    then solves the strictly convex quadratic program.
    """

    lam = np.sort(np.asarray(lambda_spectrum, dtype=float).reshape(-1))[::-1]
    if not lam.size or not np.all(np.isfinite(lam)):
        raise ValueError("The spectrum must be a nonempty finite vector.")
    mult = tuple(int(m) for m in multiplicities)
    if any(m <= 0 for m in mult) or sum(mult) != lam.size:
        raise ValueError(
            "Multiplicities must be positive and sum to the spectrum size."
        )

    # The problem is homogeneous. Normalizing its scale makes the optimizer's
    # absolute tolerances independent of the units used for the spectrum.
    scale = float(np.max(np.abs(lam)))
    if scale == 0.0:
        scale = 1.0
    lam = lam / scale

    groups = np.repeat(np.arange(len(mult)), mult)
    expansion = np.eye(len(mult))[groups]
    prefix = np.cumsum(expansion, axis=0)[:-1]
    order = np.eye(len(mult), k=0)[:-1] - np.eye(len(mult), k=1)[:-1]
    A = np.vstack((prefix, order))
    b = np.r_[np.cumsum(lam)[:-1], np.zeros(len(mult) - 1)]
    weights = np.asarray(mult, dtype=float)
    trace = float(lam.sum())
    bounds = [(0.0, None) if nonnegative else (None, None)] * len(mult)

    feasible = linprog(
        np.zeros(len(mult)),
        A_ub=-A,
        b_ub=-b,
        A_eq=weights[None, :],
        b_eq=[trace],
        bounds=bounds,
        method="highs",
    )
    if not feasible.success:
        raise ValueError(f"The pattern {mult} is infeasible: {feasible.message}")

    objective = lambda x: float(weights @ (x * x))
    result = minimize(
        objective,
        feasible.x,
        jac=lambda x: 2.0 * weights * x,
        method="SLSQP",
        bounds=Bounds(
            np.zeros(len(mult)) if nonnegative else np.full(len(mult), -np.inf),
            np.full(len(mult), np.inf),
        ),
        constraints=[
            {
                "type": "ineq",
                "fun": lambda x: A @ x - b,
                "jac": lambda _x: A,
            },
            {
                "type": "eq",
                "fun": lambda x: float(weights @ x - trace),
                "jac": lambda _x: weights,
            },
        ],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )

    reduced_scaled = np.asarray(result.x)
    mu_scaled = _expand(reduced_scaled, mult)
    # Majorization includes the prefix inequalities and equality of total sums.
    prefix_error = max(
        0.0,
        float(
            -np.min(
                np.cumsum(mu_scaled)[:-1] - np.cumsum(lam)[:-1],
                initial=0.0,
            )
        ),
    )
    majorization_error = max(
        prefix_error, abs(float(mu_scaled.sum() - lam.sum()))
    )
    order_error = max(0.0, float(np.max(np.diff(mu_scaled), initial=0.0)))
    positivity_error = (
        max(0.0, float(-np.min(mu_scaled, initial=0.0)))
        if nonnegative
        else 0.0
    )
    error = max(majorization_error, order_error, positivity_error)
    certified = result.success
    if error <= tolerance and not certified:
        # SLSQP sometimes rejects a feasible optimum during its line search.
        # For a convex problem, the first-order condition is another small LP.
        gradient = 2.0 * weights * reduced_scaled
        check = linprog(
            gradient,
            A_ub=-A,
            b_ub=-b,
            A_eq=weights[None, :],
            b_eq=[trace],
            bounds=bounds,
            method="highs",
        )
        gap = float(gradient @ reduced_scaled - check.fun) if check.success else np.inf
        certified = gap <= tolerance * max(1.0, abs(objective(reduced_scaled)))
    if error > tolerance or not certified:
        raise RuntimeError(f"QP failed validation: {result.message}.")
    reduced = reduced_scaled * scale
    mu = mu_scaled * scale
    return QPResult(reduced, mu, mult, objective(reduced))


def solve_spectrum(
    lambda_spectrum: Sequence[float], *, nonnegative: bool = False
) -> dict[str, QPResult]:
    """Infer d and solve every dimension-appropriate plateau family."""

    lam, d = _spectrum(lambda_spectrum)
    results = {}
    for name, pattern in plateau_patterns(d).items():
        try:
            results[name] = solve_majorizer(
                lam, pattern, nonnegative=nonnegative
            )
        except ValueError:
            # Some plateau families are incompatible with nonnegativity.
            if not nonnegative:
                raise
    if not results:
        raise ValueError("No dimension-appropriate plateau family is feasible.")
    return results


def joint_partial_trace_spectrum(mu: Sequence[float]) -> np.ndarray:
    """Both partial-trace spectra of ``diag(mu^downarrow)``, decreasingly sorted."""

    values, d = _spectrum(mu)
    grid = values.reshape(d, d)
    return np.sort(np.r_[grid.sum(axis=0), grid.sum(axis=1)])[::-1]


def schatten_power_bound(mu: Sequence[float], p: float) -> float:
    r"""Return ``||tr_1 C||_p^p + ||tr_2 C||_p^p`` from a majorizer."""

    if not np.isfinite(p) or p < 1:
        raise ValueError("p must be finite and at least 1.")
    return float(np.sum(np.abs(joint_partial_trace_spectrum(mu)) ** p))


def schatten_bounds(results: dict[str, QPResult], p: float) -> dict[str, float]:
    """Evaluate all candidate majorizers at one p."""

    return {
        name: schatten_power_bound(result.spectrum, p)
        for name, result in results.items()
    }


def reference_p_bounds(lambda_spectrum: Sequence[float], p: float) -> dict[str, float]:
    """The two density-matrix comparison bounds used in the paper."""

    lam, d = _spectrum(lambda_spectrum)
    if lam.min() < -1e-10 or not np.isclose(lam.sum(), 1.0, atol=1e-9):
        raise ValueError("Reference comparisons require a density spectrum.")
    if not np.isfinite(p) or p < 1:
        raise ValueError("p must be finite and at least 1.")
    power = float(np.sum(np.clip(lam, 0.0, None) ** p))
    return {"Audenaert": 1.0 + power, "Rastegin": 2.0 * d ** (p - 1) * power}


def renyi_lower_bound(
    lambda_spectrum: Sequence[float], mu: Sequence[float], alpha: float
) -> float:
    r"""Evaluate the Renyi bound using a nonnegative candidate from the solver."""

    lam, d_lam = _spectrum(lambda_spectrum)
    candidate, d_candidate = _spectrum(mu)
    if d_lam != d_candidate:
        raise ValueError("lambda and mu must describe systems of the same dimension.")
    if (
        not np.isfinite(alpha)
        or alpha <= 1
        or lam.min() < -1e-10
        or candidate.min() < -1e-10
    ):
        raise ValueError("The Renyi bound requires alpha > 1 and nonnegative spectra.")
    if not np.isclose(lam.sum(), 1.0, atol=1e-9):
        raise ValueError("The Renyi bound requires a trace-one input spectrum.")
    if not np.isclose(candidate.sum(), lam.sum(), atol=1e-8) or np.any(
        np.cumsum(candidate)[:-1] < np.cumsum(lam)[:-1] - 1e-8
    ):
        raise ValueError("mu must majorize lambda.")
    joint_power = np.sum(
        np.clip(joint_partial_trace_spectrum(candidate), 0.0, None) ** alpha
    )
    input_power = np.sum(np.clip(lam, 0.0, None) ** alpha)
    return float(np.log(joint_power / input_power) / (1.0 - alpha))


def dimension_renyi_bound(lambda_spectrum: Sequence[float]) -> float:
    """Dimension-only lower bound for an equal bipartite system."""

    _, d = _spectrum(lambda_spectrum)
    return float(-np.log(d))
