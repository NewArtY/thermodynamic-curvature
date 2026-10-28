"""Pion-gas curvature in physical units (Sec. 4.3 numbers). g_s = 3, single species, conserved number."""
import math
from thermogeom import Stat, curvature_np

HBARC = 197.327  # MeV fm
mpi = 138.0
g = 3
BE = Stat("bg", -1)
for T, mu in [(150.0, 0.0), (138.0, 0.0), (138.0, 134.0), (138.0, 137.0), (138.0, 137.8)]:
    zeta = mpi / T
    alpha = (mpi - mu) / T
    Rhat = curvature_np(zeta, alpha, BE)
    T3 = (T / HBARC) ** 3  # fm^-3
    R = 2 * math.pi ** 2 / (g * T3) * Rhat
    print(f"T={T:6.1f} mu={mu:6.1f} zeta={zeta:.3f} alpha={alpha:.4g} Rhat={Rhat:.4f} R={R:.3f} fm^3")
