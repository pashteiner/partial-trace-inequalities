# Quadratic-program bounds for partial traces

This code accompanies the paper [arXiv:2601.14158](https://arxiv.org/abs/2601.14158) and serves two purposes:

1. reproduce its numerical computations;
2. compute bounds for arbitrary spectra of equal bipartite systems.

## Quick start

```python
import numpy as np
from partial_trace_qp import solve_spectrum, schatten_bounds

lam = np.array([15, 10, 5, 4, 3, 3, 2, 2, 1]) / 45
results = solve_spectrum(lam)  # infers d and solves every admissible family
print(schatten_bounds(results, p=2))
```

For density-matrix entropy calculations, use
`solve_spectrum(lam, nonnegative=True)`.

## Automatic plateau selection

The spectrum must contain exactly `d**2` entries for an integer `d >= 3`.
The code sorts it in decreasing order, infers `d`, and generates the following
plateau starts:

| Dimension | Plateau starts | Plateau length |
|---|---:|---:|
| `d = 3` | 1, 2, 3, 4 | 6 |
| `d = 4` | 2, 3, 4 | 12 |
| `d = 5` | 3, 4 | 20 |
| `d >= 6` | 4 | `d**2 - 6` |

The subscript records the plateau start: `mu_I`, `mu_II`, `mu_III`, and
`mu_IV` have plateaus beginning at entries 1, 2, 3, and 4, respectively.
Only the dimension-appropriate names are returned.

The notebook `Partial_trace_QPs.ipynb` reproduces the plots and demonstrates both a
small-dimensional family and the automatic higher-dimensional calculation. Running
it also refreshes publication-ready PDF and PNG plots in the `figures` directory.

## Requirements

Python 3.10 or newer with NumPy, SciPy, Matplotlib, and Jupyter.
