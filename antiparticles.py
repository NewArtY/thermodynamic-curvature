# -*- coding: utf-8 -*-
r"""
antiparticles.py -- thermodynamic curvature of a relativistic ideal gas WITH antiparticles
=========================================================================================

Extension of thermogeom.py (same conventions; see its docstring).

MODEL
-----
Particle and antiparticle: same mass m and degeneracy g, opposite charge, chemical potentials
+mu and -mu.  The conserved quantity is the net charge Q = N - Nbar.  With gamma = -beta mu,

    phi_tot(beta, gamma) = phi(beta, gamma) + phi(beta, -gamma),

a 2D Hessian geometry in (beta, gamma);  -d_gamma phi_tot = (N - Nbar)/V.
Bosons require |mu| < m.  Rhat = (g T^3 / 2 pi^2) R, geometer's sign (2-sphere > 0).

NUMERICS
--------
Each species is evaluated with thermogeom.moments_np / moments_mp in its OWN kinetic chart
(beta, alpha_s), alpha_s = s gamma + beta m (s = +1 particle, s = -1 antiparticle), with the
plain third row (red='none').  The second- and third-derivative tensors are then pulled back
to the common chart (beta, gamma) by the affine Jacobian
    d_beta|gamma = d_beta|alpha + m d_alpha,   d_gamma = s d_alpha,
summed, and inserted in the plain 3x3 determinant.  The cancellation-free reduction of
thermogeom does NOT apply (the sum has no affine homothety); accuracy is checked against the
mpmath path (plain determinant, 50 digits) in main().

CLOSED FORMS (derived in verify_antiparticles_symbolic.py)
------------
* MJ (a = 0), any mu:  phi = 2 cosh(gamma) A(beta), A = int rho e^{-beta eps} = m^2 K_2(beta m)/beta
      R = -cosh(gamma) A'(A A''' - A' A'') / (4 [cosh^2 A A'' - sinh^2 A'^2]^2)
        = -(1 - t^2) <e>(<e^3> - <e><e^2>) / (2 n_tot [<e^2> - t^2 <e>^2]^2),  t = tanh(mu/T),
  <e^k> = raw MJ energy moments, n_tot = n + nbar.  NOT flat; R < 0 for every (T, mu, m).
* mu = 0, ANY statistics, massless:  R = -1/((D+1) phi) = -T/(4 P)       (D = 3).
* mu = 0, any statistics, massive:   R = -V1 (W0 V3 - W2 V1) / (4 W0^2 W2^2),
      W_k = int rho eps^k w,  V_k = int rho eps^k w'  (one species, total energy eps).
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np

import thermogeom as tg

KEYS2 = ("bb", "bg", "gg")
KEYS3 = ("bbb", "bbg", "bgg", "ggg")


# =============================================================================
# 1. one species, derivatives of phi in the common chart (beta, gamma)
# =============================================================================
def _pullback(I2, I1, I0, J3, J2, J1, J0, m, s):
    """Kinetic-chart derivatives (f_bb=I2, f_ba=I1, f_aa=I0, f_bbb=J3, ..., f_aaa=J0) of a
    species with alpha = s*gamma + beta*m  ->  derivatives in (beta, gamma)."""
    return dict(
        bb=I2 + 2 * m * I1 + m * m * I0,
        bg=s * (I1 + m * I0),
        gg=I0,
        bbb=J3 + 3 * m * J2 + 3 * m * m * J1 + m ** 3 * J0,
        bbg=s * (J2 + 2 * m * J1 + m * m * J0),
        bgg=J1 + m * J0,
        ggg=s * J0,
    )


def species_derivs_np(beta, gamma, m, st, s, D=3, **kw):
    """float64 derivatives of phi_s(beta, gamma) = phi(beta, s*gamma) (reduced, int p^{D-1} F dp)."""
    alpha = s * gamma + beta * m
    if st.a == -1 and alpha <= 0:
        raise ValueError("Bose gas requires |mu| < m (alpha = %g)" % alpha)
    mo = tg.moments_np(beta, alpha, m, st, D=D, red="none", **kw)
    # undo the internal rescaling of thermogeom (BG: by ztilde; deformed: by exp(lnscale))
    sc = math.exp(-alpha) if st.kind == "bg" else math.exp(mo["lnscale"])
    I2, I1, I0 = (sc * v for v in mo["row1"])
    J3, J2, J1 = (sc * v for v in mo["row2"])
    J0 = sc * mo["row3"][2]
    return _pullback(I2, I1, I0, J3, J2, J1, J0, m, s)


def species_derivs_mp(beta, gamma, m, st, s, D=3, dps=50, **kw):
    """mpmath derivatives of phi_s (unscaled; thermogeom's mp kernels are not rescaled)."""
    with mp.workdps(dps + 10):
        beta, gamma, m = mp.mpf(beta), mp.mpf(gamma), mp.mpf(m)
        alpha = s * gamma + beta * m
        if st.a == -1 and alpha <= 0:
            raise ValueError("Bose gas requires |mu| < m")
        mo = tg.moments_mp(beta, alpha, m, st, D=D, red="none", dps=dps, **kw)
        I2, I1, I0 = mo["row1"]
        J3, J2, J1 = mo["row2"]
        J0 = mo["row3"][2]
        return _pullback(I2, I1, I0, J3, J2, J1, J0, m, s)


# =============================================================================
# 2. curvature
# =============================================================================
def curvature_from_derivs(d):
    """R = -det[...]/(2 detg^2) from a dict of second/third derivatives in (beta, gamma).
    Returns (R, detg, det)."""
    r1 = [d["bb"], d["bg"], d["gg"]]
    r2 = [d["bbb"], d["bbg"], d["bgg"]]
    r3 = [d["bbg"], d["bgg"], d["ggg"]]
    det = tg._det3(r1, r2, r3)
    G = r1[0] * r1[2] - r1[1] * r1[1]
    return -det / (2 * G * G), G, det


def _add(d1, d2):
    return {k: d1[k] + d2[k] for k in KEYS2 + KEYS3}


def curvature_pm(zeta, x, st, path="np", D=3, dps=50, antiparticles=True, **kw):
    """Reduced curvature Rhat at T = 1 (beta = 1, m = zeta), x = mu/T  (gamma = -x).

    antiparticles=False gives the single-species value (cross-check with thermogeom)."""
    gamma = -x
    if path == "np":
        dp = species_derivs_np(1.0, gamma, zeta, st, +1, D=D, **kw)
        if not antiparticles:
            return curvature_from_derivs(dp)[0]
        if x == 0:     # identical moments for both species: reuse, odd entries cancel exactly
            da = {k: (-v if k in ("bg", "bbg", "ggg") else v) for k, v in dp.items()}
        else:
            da = species_derivs_np(1.0, gamma, zeta, st, -1, D=D, **kw)
        return curvature_from_derivs(_add(dp, da))[0]
    with mp.workdps(dps + 10):
        dp = species_derivs_mp(1, gamma, zeta, st, +1, D=D, dps=dps, **kw)
        if not antiparticles:
            return +curvature_from_derivs(dp)[0]
        if x == 0:
            da = {k: (-v if k in ("bg", "bbg", "ggg") else v) for k, v in dp.items()}
        else:
            da = species_derivs_mp(1, gamma, zeta, st, -1, D=D, dps=dps, **kw)
        return +curvature_from_derivs(_add(dp, da))[0]


def curvature_mu0_symmetric_mp(zeta, st, D=3, dps=50):
    """Independent mu = 0 formula  Rhat = -V1 (W0 V3 - W2 V1) / (4 W0^2 W2^2)  (total-energy moments),
    evaluated with the m-expansion written out so that no explicit eps-moment is formed."""
    with mp.workdps(dps + 10):
        d = species_derivs_mp(1, 0, zeta, st, +1, D=D, dps=dps)
        W0, W2 = d["gg"], d["bb"]
        V1, V3 = d["bgg"], d["bbb"]
        return +(-V1 * (W0 * V3 - W2 * V1) / (4 * W0 ** 2 * W2 ** 2))


# =============================================================================
# 3. closed forms
# =============================================================================
def A_total(beta, m, D=3):
    """A(beta) = int p^{D-1} e^{-beta eps} dp = m^2 K_2(beta m)/beta  (D = 3; mpmath)."""
    if D != 3:
        raise NotImplementedError
    return m * m * mp.besselk(2, beta * m) / beta


def R_MJ_pm(zeta, x, dps=50):
    """Exact MJ curvature with antiparticles, D = 3, T = 1, x = mu/T:
       R = -cosh(x) A'(A A''' - A' A'') / (4 [cosh^2 A A'' - sinh^2 A'^2]^2)."""
    with mp.workdps(dps + 10):
        zeta, x = mp.mpf(zeta), mp.mpf(x)
        if zeta == 0:
            A = [mp.mpf(2), mp.mpf(-6), mp.mpf(24), mp.mpf(-120)]      # 2 beta^-3 at beta = 1
        else:
            A = [mp.diff(lambda bb: A_total(bb, zeta), 1, k) for k in range(4)]
        C, S = mp.cosh(x), mp.sinh(x)
        R = -C * A[1] * (A[0] * A[3] - A[1] * A[2]) / (4 * (C * C * A[0] * A[2] - S * S * A[1] ** 2) ** 2)
        return +R


def R_MJ_pm_moments(zeta, x, dps=50):
    """Same via MJ raw energy moments (cumulants of thermogeom.mj_cumulants):
       Rhat = -(1-t^2) m1 (m3 - m1 m2) / (2 nhat_tot (m2 - t^2 m1^2)^2),  nhat_tot = 2 cosh(x) A."""
    with mp.workdps(dps + 10):
        k1, k2, k3, _ = tg.mj_cumulants(zeta, 3, dps + 10)
        m1, m2, m3 = k1, k2 + k1 ** 2, k3 + 3 * k1 * k2 + k1 ** 3
        A = mp.mpf(2) if zeta == 0 else A_total(1, mp.mpf(zeta))
        t = mp.tanh(x)
        ntot = 2 * mp.cosh(x) * A
        return +(-(1 - t * t) * m1 * (m3 - m1 * m2) / (2 * ntot * (m2 - t * t * m1 ** 2) ** 2))


def Delta_dilute(zeta, dps=50):
    """O(e^{-zeta}) quantum correction at mu = 0:  Rhat = R_MJpm + a Delta + O(e^{-zeta} ...).
    Delta from verify_antiparticles_symbolic.py (A4) with b_k = d^k/dbeta^k A(2 beta) at beta = 1."""
    with mp.workdps(dps + 10):
        zeta = mp.mpf(zeta)
        a0, a1, a2, a3 = [mp.diff(lambda bb: A_total(bb, zeta), 1, k) for k in range(4)]
        b0, b1, b2, b3 = [mp.diff(lambda bb: A_total(2 * bb, zeta), 1, k) for k in range(4)]
        num = (a0 ** 2 * a1 * a2 * b3 - 2 * a0 ** 2 * a1 * a3 * b2 + 4 * a0 ** 2 * a2 * a3 * b1
               + a0 * a1 ** 2 * a2 * b2 - 8 * a0 * a1 * a2 ** 2 * b1 - 4 * a0 * a1 * a2 * a3 * b0
               + 8 * a1 ** 2 * a2 ** 2 * b0)
        return +(num / (8 * a0 ** 3 * a2 ** 3))


def R_MJ_pm_NR(zeta):
    """Large-zeta series:  -(3/4) sqrt(2/pi) zeta^{-7/2} e^{zeta} [1 - 11/(8z) - 1311/(128 z^2) + 33615/(1024 z^3)]."""
    z = zeta
    return (-0.75 * math.sqrt(2 / math.pi) * z ** -3.5 * math.exp(z)
            * (1 - 11 / (8 * z) - 1311 / (128 * z * z) + 33615 / (1024 * z ** 3)))


def Delta_NR(zeta):
    """Large-zeta series:  Delta = -(3/(8 sqrt(pi))) zeta^{-5/2} [1 - 41/(16 z) - 2439/(512 z^2)]."""
    z = zeta
    return -3 / (8 * math.sqrt(math.pi)) * z ** -2.5 * (1 - 41 / (16 * z) - 2439 / (512 * z * z))


def R_massless_mu0(a):
    """mu = 0, m = 0:  Rhat = -1/(4 phihat_tot);  FD: -45/(7 pi^4), BE: -45/(8 pi^4), MJ: -1/16."""
    if a == 0:
        return -1.0 / 16
    return -45.0 / ((7.0 if a == 1 else 8.0) * math.pi ** 4)


def R_massless_FD_pm(x):
    """Massless FD with antiparticles, any x = mu/T (exact; phi = 2 beta^-3 [7pi^4/360 + pi^2 g^2/12 + g^4/24])."""
    p2 = math.pi ** 2
    return (-45 * (15 * x ** 6 - 3 * p2 * x ** 4 + 21 * p2 ** 2 * x ** 2 + 7 * p2 ** 3)
            / (p2 * (15 * x ** 4 + 6 * p2 * x ** 2 + 7 * p2 ** 2) ** 2))


def R_massless_MJ_pm(x):
    """Massless MJ with antiparticles: Rhat = -cosh x / (cosh^2 x + 3)^2."""
    c = math.cosh(x)
    return -c / (c * c + 3) ** 2
