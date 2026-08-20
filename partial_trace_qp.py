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
    return {
        f"mu_{before + 1}": (1,) * before
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
    mult = tuple(int(m) for m in multiplicities)
    if any(m <= 0 for m in mult) or sum(mult) != lam.size:
        raise ValueError("Multiplicities must be positive and sum to the spectrum size.")

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

    reduced = np.asarray(result.x)
    mu = _expand(reduced, mult)
    # Majorization includes the prefix inequalities and equality of total sums.
    prefix_error = max(
        0.0,
        float(-np.min(np.cumsum(mu)[:-1] - np.cumsum(lam)[:-1], initial=0.0)),
    )
    majorization_error = max(prefix_error, abs(float(mu.sum() - lam.sum())))
    order_error = max(0.0, float(np.max(np.diff(mu), initial=0.0)))
    positivity_error = (
        max(0.0, float(-np.min(mu, initial=0.0))) if nonnegative else 0.0
    )
    if not result.success or max(
        majorization_error, order_error, positivity_error
    ) > tolerance:
        raise RuntimeError(f"QP failed validation: {result.message}.")
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
    """Spectrum of both partial traces of diag(mu), in decreasing order."""

    values, d = _spectrum(mu)
    grid = values.reshape(d, d)
    return np.sort(np.r_[grid.sum(axis=0), grid.sum(axis=1)])[::-1]


def schatten_power_bound(mu: Sequence[float], p: float) -> float:
    r"""Return ``||tr_1 C||_p^p + ||tr_2 C||_p^p`` from a majorizer."""

    if p < 1:
        raise ValueError("p must be at least 1.")
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
    if p < 1:
        raise ValueError("p must be at least 1.")
    power = float(np.sum(lam**p))
    return {"Audenaert": 1.0 + power, "Rastegin": 2.0 * d ** (p - 1) * power}


def renyi_lower_bound(
    lambda_spectrum: Sequence[float], mu: Sequence[float], alpha: float
) -> float:
    r"""Lower bound for ``S_a(rho_A)+S_a(rho_B)-S_a(rho_AB)``."""

    lam, _ = _spectrum(lambda_spectrum)
    candidate, _ = _spectrum(mu)
    if alpha <= 1 or lam.min() < -1e-10 or candidate.min() < -1e-10:
        raise ValueError("The Renyi bound requires alpha > 1 and nonnegative spectra.")
    joint_power = np.sum(joint_partial_trace_spectrum(candidate) ** alpha)
    input_power = np.sum(lam**alpha)
    return float(np.log(joint_power / input_power) / (1.0 - alpha))


def dimension_renyi_bound(lambda_spectrum: Sequence[float]) -> float:
    """Dimension-only lower bound for an equal bipartite system."""

    _, d = _spectrum(lambda_spectrum)
    return float(-np.log(d))
