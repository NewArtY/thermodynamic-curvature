# Code for "Thermodynamic curvature of relativistic ideal gases: flatness, the second virial coefficient and deformed statistics"

N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, Qing-Hua Qin (submitted to *Nuclear Physics B*).

This package computes every number, table and figure in the article. All results are in the sphere-positive curvature convention, in which the ideal Fermi–Dirac (FD) gas has R < 0 and the ideal Bose–Einstein (BE) gas has R > 0. The reduced curvature is R̂ = g_s T³ R / (2π²).

## Requirements

- Python ≥ 3.11
- numpy, scipy, mpmath, sympy, matplotlib (tested versions are pinned in `requirements.txt`)

```
pip install -r requirements.txt
```

## Contents

### Core modules

| File | Purpose |
|---|---|
| `thermogeom.py` | Core module. Double-exponential momentum quadrature with a fast float64 path and an arbitrary-precision mpmath path; cancellation-free curvature determinant; closed forms for r₀, Ξ, B₂, Υ, Γ_B, 𝒞(q) and 𝒦(κ); root finding for the Kaniadakis crossover z̃*. |
| `antiparticles.py` | Particle–antiparticle gas, φ(β,γ) + φ(β,−γ) (Sec. 4.4). |

### Figures and tables

| File | Purpose |
|---|---|
| `make_figures.py` | Figs. 1, 3 and 4. Results are cached in `data/fig_cache_*.npz`; pass `--recompute` to rebuild the cache. |
| `make_antiparticles.py` | Fig. 2 (`fig2_antiparticles`) and `data/antiparticles.json`. |
| `tables.py`, `closed_forms_print.py` | Tables and closed-form constants. |

### Checks

| File | What it checks |
|---|---|
| `verify_convention.py` | Symbolic proof that the determinant formula is the Ricci scalar (unit sphere gives +2). |
| `verify_symbolic.py` | Flatness theorem, virial formula, O(κ²) Kaniadakis curvature and the deep-dilute constants. |
| `verify_rigidity.py` | First-order polynomial deformations. |
| `verify_asymptotics.py` | Ultrarelativistic and non-relativistic series. |
| `verify_quadrature.py` | Fast vs arbitrary-precision paths, the massless polylogarithm formula, and an independent finite-difference (Brioschi) curvature. |
| `verify_bg.py` | Boltzmann–Gibbs gases (Sec. 4). |
| `verify_deformed.py` | Deformed gases (Sec. 5). |
| `verify_antiparticles_symbolic.py` | Sec. 4.4. |
| `compare_old_code.py` | Diagnostic comparison with the code of an earlier, longer version of the manuscript (archive [10.5281/zenodo.21383735](https://doi.org/10.5281/zenodo.21383735); set `OLD_CODE` to its unpacked root). It is not required to reproduce the article. |

### Data

`data/` holds the JSON, NPZ and log outputs of all scripts.

## Usage

Run every script from this directory, for example `python make_figures.py`. Running times range from seconds to a few minutes per script. Figures are written to `../manuscript/figures/` if that folder exists, otherwise to `./figures/`.

Additional checks: `pion_numbers.py` (pion-gas numbers of Sec. 4.3) and `mixture_sign_check.py` (nucleon–pion mixture example of Sec. 4.4).

## Citation

Please cite the article and this archived release (see `CITATION.cff`; the Zenodo DOI is added on release).

## Licence

MIT (see `LICENSE`).
