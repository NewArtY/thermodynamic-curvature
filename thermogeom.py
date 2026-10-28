# -*- coding: utf-8 -*-
r"""
thermogeom.py -- thermodynamic (Fisher--Rao / Ruppeiner) geometry of the
relativistic ideal gas with Boltzmann--Gibbs, Tsallis and Kaniadakis occupations
===============================================================================

Reproducibility module for the NPB manuscript.  Written from scratch; it does
NOT import anything from the old Annals-of-Physics code.

MODEL
-----
Single species, ideal gas, D spatial dimensions (default D = 3), internal
degeneracy g, conserved particle number (no antiparticles), c = hbar = k_B = 1.

    eps_p = sqrt(p^2 + m^2),  zeta = m/T,  z = e^{mu/T},
    ztilde = e^{(mu - m)/T} = z e^{-zeta}  in (0, 1].

Coordinates on the state space: x = (beta, gamma), beta = 1/T, gamma = -beta mu.
Argument of the occupation: y = beta(eps - mu) = beta eps + gamma.

Master occupation, a in {+1 (FD), -1 (BE), 0 (classical)}, sigma in {0, q, kappa}:

    n(y) = 1 / ( [exp_sigma(-y)]^{-1} + a ) = u / (1 + a u),   u(y) = exp_sigma(-y),

    exp_0 = exp,
    exp_q(-y)     = [1 + (q-1) y]^{-1/(q-1)}            (q > 1, power law),
    exp_kappa(-y) = exp(-arcsinh(kappa y)/kappa)       (reciprocal, power law).

Massieu potential per unit volume (Hessian potential of the metric):

    phi(beta, gamma) = ln Z / V = c_D g  int_0^inf p^{D-1} F(y) dp,   F' = -n,  F(inf) = 0,
    c_D = S_{D-1}/(2 pi)^D,  S_{D-1} = 2 pi^{D/2}/Gamma(D/2)       (D = 3: c_3 g = g/(2 pi^2)).

    d_gamma phi = -N/V,  d_beta phi = -E/V,   g_ij = d_i d_j phi.

INTERNAL COORDINATES
--------------------
All integrals are done in the affinely equivalent "kinetic" coordinates
(beta, alpha), alpha = gamma + beta m = -ln ztilde = beta (m - mu), with the
kinetic energy e = eps - m, so that y = beta e + alpha.  The map
(beta, gamma) -> (beta, alpha) is affine with unit Jacobian, hence the Hessian
metric, det g, the 3x3 determinant below and the curvature are identical in
both systems; the kinetic form removes the catastrophic cancellations
(eps^2 vs eps^2) of the non-relativistic regime and of the condensation edge.

    g_bb = int rho e^2 w,  g_ba = int rho e w,  g_aa = int rho w,      w  = F'' = -dn/dy > 0
    d_b g_.. = int rho e^{k+1} w',   d_a g_.. = int rho e^k w',          w' = F''' = dw/dy
    (rho dp = p^{D-1} dp; k = 2, 1, 0 for bb, ba, aa).

CURVATURE CONVENTION  (verified symbolically in verify_convention.py)
----------------------
R is the Ricci scalar in the geometer's (MTW) convention: unit 2-sphere R = +2,
R = 2K (K = Gaussian curvature).  For a 2D Hessian metric it equals EXACTLY

    R = - det[[phi_bb , phi_ba , phi_aa ],
              [phi_bbb, phi_bba, phi_baa],
              [phi_bba, phi_baa, phi_aaa]] / (2 (det g)^2).

Ruppeiner (RMP 67 (1995) 605; AJP 78 (2010) 1170) uses Weinberg's sign
convention, R_Rupp = -R.

NORMALISATION
-------------
Routines return the REDUCED, DIMENSIONLESS curvature  Rhat(zeta, alpha)
computed with the reduced potential  phihat = int p^{D-1} F dp  at T = 1.
The physical curvature (dimension of volume) is

    R = Rhat / (c_D g T^D)          (D = 3:  R = (2 pi^2 / g) T^{-3} Rhat).

The dilute-limit constant c1 is defined by  Rhat -> -a c1(zeta)  as ztilde -> 0
(BG statistics), i.e.  R -> -a (2 pi^2/g) T^{-3} c1(zeta)  in D = 3.

NUMERICS
--------
* Quadrature: double-exponential (exp-sinh) rule on p in (0, inf),
  p = p0 exp(pi/2 sinh t).  It resolves the near-condensation Bose peak at
  p ~ sqrt(2 m T alpha) (log-uniform node density near 0) and the algebraic
  tails of the deformed statistics (double-exponential decay in t).
  - fast path: numpy float64, fixed step h (default 1/128) -> ~1e-12 relative;
  - high-precision path: mpmath, step halving until convergence to `dps` digits.
* Cancellation-free determinant ("symmetry-reduced third row").  For an affine
  vector field X = (lam beta) d_beta + (lam alpha + b) d_alpha and a constant c',
  X(row1) = X^beta row2 + X^alpha row3, hence
      det(row1, row2, row3) = det(row1, row2, Krow) / X^alpha,
      Krow_k = int rho e^k K(y),   K(y) = (lam y + b) w'(y) - c' w(y).
  This is an exact identity for any (lam, b, c').  When phi has the affine
  symmetry X phi = c phi + affine (c' = c - 2 lam), K vanishes identically and
  R = 0 exactly; near such a symmetry K is small and is evaluated from an
  analytically simplified formula (no cancellation):
    'shift'   lam=0, b=1, c'=-1 :  K = w + w'            (BG: K = 2 a u^2/(1+au)^3)
    'dilq'    Tsallis, lam=1, b=1/(q-1), c'=-q/(q-1):     K = 2 a u^2 / ((q-1) v (1+au)^3)
    'dilk'    Kaniadakis deep-dilute, lam=1, b=0, c'=-(1+1/kappa)
    'none'    lam=0, b=1, c'=0  :  K = w'  (plain determinant, for cross-checks)
* Bose denominators use 1 - u = -expm1(ln u) (no cancellation at condensation); the
  fast path rescales the kernels by u(alpha) (BG: by ztilde) so that arbitrarily dilute
  states do not underflow.

MEASURED ACCURACY (verify_quadrature.py, verify_deformed.py)
------------------------------------------------------------
* mpmath path: dps=30 and dps=45 agree to <= 7e-38 (134 cases); independent Brioschi curvature
  from finite differences of the potential agrees to 1e-7...1e-8 (FD-limited).
* fast path vs mpmath: ~1e-14 relative for BG (including alpha = 1e-10 near condensation
  and ztilde = e^-1000), Tsallis q <= 1.2, Kaniadakis kappa <= 0.2;  degraded to ~1e-8
  (q = 1.24) and ~3e-7 (kappa = 0.24) because the self-similar heavy tail makes the 2x2
  cofactors of the determinant nearly cancel.  Use the mp path for numbers near the
  existence bounds q -> 1 + 1/(D+1), kappa -> 1/(D+1).

MAIN ENTRY POINTS
-----------------
  Stat(kind, a, par)                    statistics descriptor
  curvature_np(zeta, alpha, st, D)      fast reduced curvature Rhat (T = 1)
  curvature_mp(zeta, alpha, st, D, dps) high-precision Rhat
  R_physical(Rhat, T, g, D)             physical curvature (volume units)
  c1_closed, Xi_dilute, B2_reduced      dilute-limit closed forms  (R_0 = -2 Xi B_2)
  Y_kappa, zstar_small_kappa            O(kappa^2) Kaniadakis curvature, crossover law
  R_powerlaw_quantum                    exact NR (s=D/2) / UR (s=D) quantum-gas curvature
  zstar_root, find_sign_changes         sign-change location
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mpmath as mp
import numpy as np
from scipy import special as sps

# =============================================================================
# 0. Normalisation
# =============================================================================


def sphere_area(D):
    """Area S_{D-1} of the unit (D-1)-sphere."""
    return 2.0 * math.pi ** (D / 2.0) / math.gamma(D / 2.0)


def phase_space_prefactor(g=1.0, D=3):
    """c_D g = g S_{D-1}/(2 pi)^D :  phi = c_D g int p^{D-1} F dp   (D=3: g/(2 pi^2))."""
    return g * sphere_area(D) / (2.0 * math.pi) ** D


def R_physical(Rhat, T, g=1.0, D=3):
    """Physical curvature (volume units) from the reduced dimensionless Rhat(zeta, alpha)."""
    return Rhat / (phase_space_prefactor(g, D) * T ** D)


def alpha_from_ztilde(zt):
    """alpha = -ln ztilde (use alpha directly near ztilde -> 1 to keep precision)."""
    return -math.log(zt)


# =============================================================================
# 1. Statistics
# =============================================================================


@dataclass(frozen=True)
class Stat:
    """Occupation family.

    kind : 'bg' (sigma = 0), 'q' (Tsallis), 'k' (Kaniadakis)
    a    : +1 Fermi-Dirac, -1 Bose-Einstein, 0 classical
    par  : q (kind 'q', 1 < q) or kappa (kind 'k', 0 < kappa < 1)
    """

    kind: str
    a: int
    par: float = 0.0

    def label(self):
        nm = {1: "FD", -1: "BE", 0: "cl"}[self.a]
        if self.kind == "bg":
            return "BG-" + nm
        return "%s(%s=%g)-%s" % ({"q": "Tsallis", "k": "Kaniadakis"}[self.kind],
                                 "q" if self.kind == "q" else "kappa", self.par, nm)


def default_reduction(st, alpha=None):
    """Pick the best-conditioned (exact) row reduction for the 3x3 determinant."""
    if st.kind == "bg":
        return "shift"
    if st.kind == "q":
        return "dilq"
    # Kaniadakis: shift symmetry is approximate for kappa^3 alpha^2 << 1,
    # dilation symmetry for kappa^3 alpha^2 >> 1
    k = st.par
    if alpha is not None and k ** 3 * float(alpha) ** 2 > 1.0:
        return "dilk"
    return "shift"


def _reduction_vector(st, red, beta, alpha):
    """(X^beta, X^alpha) of the affine field used by the row reduction."""
    if red in ("shift", "none"):
        return 0.0, 1.0
    if red == "dilq":
        q = st.par
        return beta, alpha + 1.0 / (q - 1.0)
    if red == "dilk":
        return beta, alpha
    raise ValueError(red)


# ---------------------------------------------------------------------------
# 1a. numpy kernels  (vectorised in y)
# ---------------------------------------------------------------------------


def _one_plus_au_np(lnu, u, a):
    """1 + a u, with 1 - u = -expm1(ln u) for bosons (no cancellation near condensation)."""
    if a == -1:
        return -np.expm1(lnu)
    return 1.0 + a * u


def kernels_np(y, st, red, uscale=1.0, alpha_bg=0.0, lnscale=0.0):
    r"""Return (n, w, w', K) at the argument(s) y (numpy arrays).

    u = exp_sigma(-y);  r = E'/E, t = E''/E with E = 1/u;
    n = u/(1+au), w = r u/(1+au)^2, w' = u (t(1+au) - 2 r^2)/(1+au)^3.

    `uscale` (BG only): the kernels are returned divided by the scale z0 where
    u = z0 * uhat (used to factor out a tiny ztilde; K is divided by z0^2).
    For BG pass y = beta*e (WITHOUT alpha), uscale = ztilde, alpha_bg = -ln ztilde.
    `lnscale` (deformed only): all four kernels are returned divided by exp(lnscale)
    (one overall power of u is rescaled; the u inside 1+au is the true one).  Then
    det -> det/s^3, G^2 -> G^2/s^4 and the true curvature is Rhat_scaled / s.
    """
    a = st.a
    if st.kind == "bg":
        uh = np.exp(-y)
        u = uscale * uh
        D1 = _one_plus_au_np(-(y + alpha_bg), u, a)
        n = uh / D1
        w = uh / D1 ** 2
        wp = uh * (a * u - 1.0) / D1 ** 3
        if red == "shift":
            K = 2.0 * a * uh * uh / D1 ** 3        # divided by uscale^2
        elif red == "none":
            K = wp / uscale                          # true w' divided by uscale^2
        else:
            raise ValueError("reduction %s not available for BG" % red)
        return n, w, wp, K
    if uscale != 1.0:
        raise ValueError("uscale only for BG")
    if st.kind == "q":
        q = st.par
        v = 1.0 + (q - 1.0) * y
        lnu = -np.log1p((q - 1.0) * y) / (q - 1.0)
        u = np.exp(lnu)
        uh = np.exp(lnu - lnscale)
        r = 1.0 / v
        t = (2.0 - q) / v ** 2
        D1 = _one_plus_au_np(lnu, u, a)
        n = uh / D1
        w = r * uh / D1 ** 2
        wp = uh * (t * D1 - 2.0 * r * r) / D1 ** 3
        if red == "dilq":
            K = 2.0 * a * uh * u / ((q - 1.0) * v * D1 ** 3)
        elif red == "shift":
            K = w + wp
        elif red == "none":
            K = wp
        else:
            raise ValueError(red)
        return n, w, wp, K
    if st.kind == "k":
        k = st.par
        ky = k * y
        s = np.sqrt(1.0 + ky * ky)
        lnu = -np.arcsinh(ky) / k
        u = np.exp(lnu)
        uh = np.exp(lnu - lnscale)
        r = 1.0 / s
        t = 1.0 / s ** 2 - k * ky / s ** 3
        D1 = _one_plus_au_np(lnu, u, a)
        n = uh / D1
        w = r * uh / D1 ** 2
        wp = uh * (t * D1 - 2.0 * r * r) / D1 ** 3
        if red == "shift":
            # r + t - 2 r^2 = kappa^2 y^2/((s+1) s^2) - kappa^2 y/s^3   (no cancellation)
            b0 = ky * ky / ((s + 1.0) * s * s) - k * ky / s ** 3
            K = uh * (b0 + a * u * (r + t)) / D1 ** 3
        elif red == "dilk":
            # y w' + (1+1/kappa) w ; a=0 part = [1 + s/(kappa(s+kappa y))]/s^3 (exact)
            c = 1.0 + 1.0 / k
            b0 = (1.0 + s / (k * (s + ky))) / s ** 3
            K = uh * (b0 + a * u * (y * t + c * r)) / D1 ** 3
        elif red == "none":
            K = wp
        else:
            raise ValueError(red)
        return n, w, wp, K
    raise ValueError(st.kind)


# ---------------------------------------------------------------------------
# 1b. mpmath kernels (scalar)
# ---------------------------------------------------------------------------


def kernels_mp(y, st, red):
    """mpmath version of kernels_np (scalar y); returns (n, w, w', K)."""
    a = st.a
    if st.kind == "bg":
        lnu = -y
        r = t = mp.mpf(1)
    elif st.kind == "q":
        q = mp.mpf(st.par)
        v = 1 + (q - 1) * y
        lnu = -mp.log1p((q - 1) * y) / (q - 1)
        r = 1 / v
        t = (2 - q) / v ** 2
    elif st.kind == "k":
        k = mp.mpf(st.par)
        ky = k * y
        s = mp.sqrt(1 + ky * ky)
        lnu = -mp.asinh(ky) / k
        r = 1 / s
        t = 1 / s ** 2 - k * ky / s ** 3
    else:
        raise ValueError(st.kind)
    u = mp.exp(lnu)
    D1 = -mp.expm1(lnu) if a == -1 else 1 + a * u
    n = u / D1
    w = r * u / D1 ** 2
    wp = u * (t * D1 - 2 * r * r) / D1 ** 3
    if red == "none":
        K = wp
    elif red == "shift":
        if st.kind == "bg":
            K = 2 * a * u * u / D1 ** 3
        elif st.kind == "k":
            b0 = ky * ky / ((s + 1) * s * s) - k * ky / s ** 3
            K = u * (b0 + a * u * (r + t)) / D1 ** 3
        else:
            K = w + wp
    elif red == "dilq":
        K = 2 * a * u * u / ((q - 1) * v * D1 ** 3)
    elif red == "dilk":
        c = 1 + 1 / k
        b0 = (1 + s / (k * (s + ky))) / s ** 3
        K = u * (b0 + a * u * (y * t + c * r)) / D1 ** 3
    else:
        raise ValueError(red)
    return n, w, wp, K


def occupation_np(y, st):
    """Mean occupation n(y) (numpy)."""
    return kernels_np(y, st, "none")[0]


def F_potential_np(y, st):
    """Single-mode potential F(y) (F' = -n, F(inf) = 0); closed forms where they exist."""
    a = st.a
    if st.kind == "bg":
        u = np.exp(-y)
        return u if a == 0 else np.log1p(a * u) / a
    if a != 0:
        raise NotImplementedError("deformed quantum F: integrate n numerically")
    if st.kind == "q":
        q = st.par
        return np.power(1.0 + (q - 1.0) * y, (q - 2.0) / (q - 1.0)) / (2.0 - q)
    if st.kind == "k":
        k = st.par
        u = np.exp(-np.arcsinh(k * y) / k)
        return u * (np.sqrt(1.0 + (k * y) ** 2) + k * k * y) / (1.0 - k * k)
    raise ValueError(st.kind)


# =============================================================================
# 2. Quadrature on p in (0, inf):  exp-sinh  p = p0 exp(pi/2 sinh t)
# =============================================================================

_NODE_CACHE = {}


def _de_nodes_np(h, tlo, thi):
    key = (h, tlo, thi)
    if key not in _NODE_CACHE:
        t = np.arange(-math.ceil(tlo / h), math.ceil(thi / h) + 1) * h
        x = 0.5 * math.pi * np.sinh(t)          # ln(p/p0)
        jac = 0.5 * math.pi * np.cosh(t)         # d ln p / dt
        _NODE_CACHE[key] = (x, jac)
    return _NODE_CACHE[key]


def _p_scale(beta, alpha, m, st):
    """Characteristic momentum of the integrand (centre of the exp-sinh map)."""
    if st.kind == "bg":
        es = 1.0 / beta
    elif st.kind == "q":
        es = (1.0 + (st.par - 1.0) * max(alpha, 0.0)) / beta
    else:
        es = math.sqrt(1.0 + (st.par * max(alpha, 0.0)) ** 2) / beta
    return math.sqrt(es * es + 2.0 * m * es)


def moments_np(beta, alpha, m, st, D=3, red=None, h=1.0 / 128, tlo=4.2, thi=None, p0=None):
    """Fast moments.  Returns dict(row1, row2, row3, Krow, Xb, Xa, G, scale).

    row1 = (g_bb, g_ba, g_aa); row2 = d_beta row1; row3 = d_alpha row1; Krow as in module doc.
    For BG all rows are divided by ztilde (Krow by ztilde^2) -- R is unaffected.
    """
    if red is None:
        red = default_reduction(st, alpha)
    if thi is None:
        thi = 3.6 if st.kind == "bg" else 5.6
    if p0 is None:
        p0 = _p_scale(beta, alpha, m, st)
    x, jac = _de_nodes_np(h, tlo, thi)
    p = p0 * np.exp(x)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        e = p * p / (np.sqrt(p * p + m * m) + m)          # kinetic energy, no cancellation
        W = h * jac * p ** D                               # p^{D-1} dp = p^D d(ln p)
        lnscale = 0.0
        if st.kind == "bg":
            zt = math.exp(-alpha)
            n, w, wp, K = kernels_np(beta * e, st, red, uscale=zt, alpha_bg=alpha)
        else:
            lnscale = lnu_of_y(alpha, st)            # ln u at the band edge (p = 0)
            n, w, wp, K = kernels_np(beta * e + alpha, st, red, lnscale=lnscale)
        ok = np.isfinite(W) & np.isfinite(w) & np.isfinite(wp) & np.isfinite(K) & np.isfinite(e)
        W = np.where(ok, W, 0.0)
        e = np.where(ok, e, 0.0)
        w = np.where(ok, w, 0.0)
        wp = np.where(ok, wp, 0.0)
        K = np.where(ok, K, 0.0)
        Ww, Wwp, WK = W * w, W * wp, W * K
        e2 = e * e
        I0, I1, I2 = Ww.sum(), (Ww * e).sum(), (Ww * e2).sum()
        J0, J1, J2, J3 = Wwp.sum(), (Wwp * e).sum(), (Wwp * e2).sum(), (Wwp * e2 * e).sum()
        K0, K1, K2 = WK.sum(), (WK * e).sum(), (WK * e2).sum()
    Xb, Xa = _reduction_vector(st, red, beta, alpha)
    return dict(row1=np.array([I2, I1, I0]), row2=np.array([J3, J2, J1]),
                row3=np.array([J2, J1, J0]), Krow=np.array([K2, K1, K0]),
                Xb=Xb, Xa=Xa, G=I0 * I2 - I1 * I1, red=red, lnscale=lnscale)


def lnu_of_y(y, st):
    """ln exp_sigma(-y) (scalar)."""
    if st.kind == "bg":
        return -y
    if st.kind == "q":
        return -math.log1p((st.par - 1.0) * y) / (st.par - 1.0)
    return -math.asinh(st.par * y) / st.par


def _det3(r1, r2, r3):
    return (r1[0] * (r2[1] * r3[2] - r2[2] * r3[1])
            - r1[1] * (r2[0] * r3[2] - r2[2] * r3[0])
            + r1[2] * (r2[0] * r3[1] - r2[1] * r3[0]))


def curvature_np(zeta, alpha, st, D=3, red=None, beta=1.0, **kw):
    """Reduced curvature Rhat (T = 1/beta, m = zeta/beta) -- fast float64 path.

    Returns Rhat such that R_phys = Rhat / (c_D g T^D) when beta = 1.
    """
    m = zeta / beta
    mo = moments_np(beta, alpha, m, st, D=D, red=red, **kw)
    det = _det3(mo["row1"], mo["row2"], mo["Krow"]) / mo["Xa"]
    Rs = -det / (2.0 * mo["G"] ** 2)
    if mo["lnscale"] == 0.0:
        return Rs
    with np.errstate(over="ignore"):
        return Rs * math.exp(-mo["lnscale"]) if -mo["lnscale"] < 709.0 else Rs * math.inf


def curvature_plain_np(zeta, alpha, st, D=3, beta=1.0, **kw):
    """Same as curvature_np but with the plain (unreduced) third row (cross-check)."""
    return curvature_np(zeta, alpha, st, D=D, red="none", beta=beta, **kw)


# ---------------------------------------------------------------------------
# 2b. mpmath high-precision path
# ---------------------------------------------------------------------------


def tail_index(st):
    """P such that n ~ y^{-P} at large y (inf for BG)."""
    if st.kind == "bg":
        return math.inf
    return 1.0 / (st.par - 1.0) if st.kind == "q" else 1.0 / st.par


def _thi_mp(st, D, dps):
    """Upper t-limit so that the slowest moment tail p^{D+1-P} (in d ln p) is < 10^-dps."""
    if st.kind == "bg":
        return 4.0
    delta = tail_index(st) - D - 1.0          # decay rate in ln p of g_bb and phi_bbb integrands
    if delta <= 0:
        raise ValueError("metric does not exist: P = %g <= D + 1" % tail_index(st))
    xmax = (dps * math.log(10.0) + 30.0) / delta + 50.0
    return min(max(math.asinh(xmax / (0.5 * math.pi)), 5.0), 10.0)


def moments_mp(beta, alpha, m, st, D=3, red=None, dps=40, tlo=None, thi=None,
               h0=None, maxlev=9, tol=None, p0=None, verbose=False):
    """High-precision moments via exp-sinh with step halving.

    Returns dict(row1, row2, row3, Krow, Xb, Xa, G, levels, relchange).
    The reduced and plain third rows are both returned (Krow uses `red`).
    """
    with mp.workdps(dps + 10):
        beta = mp.mpf(beta)
        alpha = mp.mpf(alpha)
        m = mp.mpf(m)
        if red is None:
            red = default_reduction(st, float(alpha))
        if p0 is None:
            p0 = mp.mpf(_p_scale(float(beta), float(alpha), float(m), st))
        if tlo is None:
            tlo = 4.6
        if thi is None:
            thi = _thi_mp(st, D, dps)
        if h0 is None:
            h0 = mp.mpf(1) / 8
        if tol is None:
            tol = mp.mpf(10) ** (-(dps - 3))
        halfpi = mp.pi / 2

        def node(t):
            x = halfpi * mp.sinh(t)
            p = p0 * mp.exp(x)
            jac = halfpi * mp.cosh(t)
            e = p * p / (mp.sqrt(p * p + m * m) + m)
            y = beta * e + alpha
            n, w, wp, K = kernels_mp(y, st, red)
            W = jac * p ** D
            e2 = e * e
            Ww, Wwp, WK = W * w, W * wp, W * K
            return [Ww * e2, Ww * e, Ww, Wwp * e2 * e, Wwp * e2, Wwp * e, Wwp, WK * e2, WK * e, WK]

        nlo = int(mp.ceil(tlo / h0))
        nhi = int(mp.ceil(thi / h0))
        S = [mp.mpf(0)] * 10
        for j in range(-nlo, nhi + 1):
            v = node(j * h0)
            S = [S[i] + v[i] for i in range(10)]
        h = h0
        est_prev = [s * h for s in S]
        rel = None
        for lev in range(1, maxlev + 1):
            h = h / 2
            # new nodes: odd multiples of h
            jlo = -2 * nlo
            jhi = 2 * nhi
            for j in range(jlo + 1, jhi, 2):
                v = node(j * h)
                S = [S[i] + v[i] for i in range(10)]
            nlo *= 2
            nhi *= 2
            est = [s * h for s in S]
            rel = max(abs(est[i] - est_prev[i]) / (abs(est[i]) + mp.mpf(10) ** (-2 * dps))
                      for i in range(10))
            if verbose:
                print("   level", lev, "h=", mp.nstr(h, 5), "rel change", mp.nstr(rel, 5))
            est_prev = est
            if rel < tol and lev >= 2:
                break
        I2, I1, I0, J3, J2, J1, J0, K2, K1, K0 = est
        Xb, Xa = _reduction_vector(st, red, beta, alpha)
        return dict(row1=[I2, I1, I0], row2=[J3, J2, J1], row3=[J2, J1, J0],
                    Krow=[K2, K1, K0], Xb=Xb, Xa=Xa, G=I0 * I2 - I1 * I1,
                    levels=lev, relchange=rel, red=red)


def curvature_mp(zeta, alpha, st, D=3, red=None, beta=1, dps=40, plain=False, **kw):
    """High-precision reduced curvature Rhat (mpf).  plain=True: unreduced determinant."""
    with mp.workdps(dps + 10):
        m = mp.mpf(zeta) / mp.mpf(beta)
        mo = moments_mp(beta, alpha, m, st, D=D, red=red, dps=dps, **kw)
        if plain:
            det = _det3(mo["row1"], mo["row2"], mo["row3"])
        else:
            det = _det3(mo["row1"], mo["row2"], mo["Krow"]) / mo["Xa"]
        R = -det / (2 * mo["G"] ** 2)
    return +R


def plain_det_scale_mp(zeta, alpha, st, D=3, beta=1, dps=40, **kw):
    """Plain determinant together with the magnitude of its largest expansion term.

    The ratio |det| / max|term| measures how close the three rows are to linear
    dependence (it is exactly 0 for an exactly flat manifold)."""
    with mp.workdps(dps + 10):
        m = mp.mpf(zeta) / mp.mpf(beta)
        mo = moments_mp(beta, alpha, m, st, D=D, red="none", dps=dps, **kw)
        r1, r2, r3 = mo["row1"], mo["row2"], mo["row3"]
        terms = [r1[0] * r2[1] * r3[2], r1[0] * r2[2] * r3[1], r1[1] * r2[0] * r3[2],
                 r1[1] * r2[2] * r3[0], r1[2] * r2[0] * r3[1], r1[2] * r2[1] * r3[0]]
        det = _det3(r1, r2, r3)
        return +det, max(abs(tt) for tt in terms), mo


# =============================================================================
# 3. Potential, density, energy (for virial / existence checks)
# =============================================================================


def potential_np(zeta, alpha, st, D=3, beta=1.0, h=1.0 / 128, tlo=4.2, thi=5.6):
    """Reduced potential phihat = int p^{D-1} F(y) dp and n = int p^{D-1} n dp, E/V (kinetic+rest)."""
    m = zeta / beta
    p0 = _p_scale(beta, alpha, m, st)
    x, jac = _de_nodes_np(h, tlo, thi)
    p = p0 * np.exp(x)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        e = p * p / (np.sqrt(p * p + m * m) + m)
        y = beta * e + alpha
        W = h * jac * p ** D
        F = F_potential_np(y, st)
        n = occupation_np(y, st)
        ok = np.isfinite(F) & np.isfinite(n) & np.isfinite(W)
        W = np.where(ok, W, 0.0)
        F = np.where(ok, F, 0.0)
        n = np.where(ok, n, 0.0)
        return (W * F).sum(), (W * n).sum(), (W * n * (e + m)).sum()


# =============================================================================
# 4. Closed forms: dilute limit (BG), second virial coefficient, O(kappa^2)
# =============================================================================
#
# Single-particle energy cumulants of the classical (Maxwell-Juttner) gas at
# coldness zeta, in units of T:  c_k = kappa_k beta^k.  With nu = (D+1)/2 and
# Rnu(zeta) = K_{nu-1}(zeta)/K_nu(zeta):
#     c_1 = D + zeta Rnu,   c_{k+1} = k c_k - zeta dc_k/dzeta,
#     dRnu/dzeta = Rnu^2 + (D/zeta) Rnu - 1.
# These are polynomials in (zeta, Rnu); built once with sympy and lambdified.

_CUMULANT_CACHE = {}


def _cumulant_polys(D):
    if D in _CUMULANT_CACHE:
        return _CUMULANT_CACHE[D]
    import sympy as sp
    zs, Rs = sp.symbols("zeta R")

    def dz(expr):
        return sp.expand(sp.diff(expr, zs) + sp.diff(expr, Rs) * (Rs ** 2 + D * Rs / zs - 1))

    c = [None, sp.expand(D + zs * Rs)]
    for k in range(1, 4):
        c.append(sp.expand(k * c[k] - zs * dz(c[k])))
    funcs = [None] + [sp.lambdify((zs, Rs), c[k], "mpmath") for k in range(1, 5)]
    _CUMULANT_CACHE[D] = (funcs, c)
    return _CUMULANT_CACHE[D]


def mj_cumulants(zeta, D=3, dps=50):
    """(c1, c2, c3, c4): dimensionless single-particle energy cumulants kappa_k beta^k (MJ gas).

    zeta = 0 returns the massless values c_k = D (k-1)!.
    """
    funcs, _ = _cumulant_polys(D)
    with mp.workdps(dps):
        zeta = mp.mpf(zeta)
        if zeta == 0:
            return tuple(mp.mpf(D) * mp.factorial(k - 1) for k in range(1, 5))
        nu = mp.mpf(D + 1) / 2
        Rn = mp.besselk(nu - 1, zeta) / mp.besselk(nu, zeta)
        return tuple(funcs[k](zeta, Rn) for k in range(1, 5))


def Xi_dilute(zeta, D=3, dps=50):
    r"""Xi(zeta) in  R_0 = -2 Xi(zeta) B_2(T)   (BG quantum gas, geometer's sign).

    Xi = c2(2z)/(4 c2(z)) + d^2/c2(z) + c3(z) d / c2(z)^2 - 1,  d = c1(z) - c1(2z)/2.
    UR: Xi -> (D+1)/4 ;  NR: Xi -> (D+2)/8.
    """
    with mp.workdps(dps):
        zeta = mp.mpf(zeta)
        c1a, c2a, c3a, _ = mj_cumulants(zeta, D, dps)
        c1b, c2b, _, _ = mj_cumulants(2 * zeta, D, dps)
        d = c1a - c1b / 2
        return c2b / (4 * c2a) + d * d / c2a + c3a * d / c2a ** 2 - 1


def B2_reduced(zeta, a, D=3, dps=50):
    r"""Second virial coefficient (P/(nT) = 1 + B_2 n + ...) in reduced units at T = 1.

    B_2 = (a/2) A(2 beta)/A(beta)^2,  A(beta) = int p^{D-1} e^{-beta eps} dp
        = C_D m^D x^{-(D-1)/2} K_nu(x), x = beta m, C_D = Gamma(D/2) 2^{(D-1)/2}/sqrt(pi).
    Physical: B_2^phys = B_2^red / (c_D g T^D).
    D = 3:  B_2 = a K_2(2 zeta) / (4 zeta^2 K_2(zeta)^2);  massless: a 2^{-(D+1)} ...
    """
    with mp.workdps(dps):
        zeta = mp.mpf(zeta)
        nu = mp.mpf(D + 1) / 2
        CD = mp.gamma(mp.mpf(D) / 2) * mp.power(2, mp.mpf(D - 1) / 2) / mp.sqrt(mp.pi)
        if zeta == 0:
            # A(beta) = Gamma(D) beta^{-D}  ->  B2 = (a/2) 2^{-D} / Gamma(D)
            return mp.mpf(a) / 2 * mp.power(2, -D) / mp.gamma(D)
        return (mp.mpf(a) / (2 * CD) * mp.power(2, -mp.mpf(D - 1) / 2) * mp.power(zeta, -nu)
                * mp.besselk(nu, 2 * zeta) / mp.besselk(nu, zeta) ** 2)


def c1_closed(zeta, D=3, dps=50):
    r"""Dilute-limit constant:  Rhat -> -a c1(zeta)  (BG quantum gas), closed form.

    c1 = 2 Xi(zeta) |B_2^red(zeta)|.   D = 3:  c1 = Xi(zeta) K_2(2 zeta)/(2 zeta^2 K_2(zeta)^2),
    UR plateau 1/16, NR tail 5/(16 sqrt(pi)) zeta^{-3/2}.
    """
    with mp.workdps(dps):
        return 2 * Xi_dilute(zeta, D, dps) * B2_reduced(zeta, 1, D, dps)


def bg_moments_old_convention(zeta, D=3, dps=50):
    r"""Old-manuscript Boltzmann moments (T = 1, m = zeta, TOTAL energy eps, p^{D-1} dp):
    M_k = int eps^k p^{D-1} e^{-eps} dp,  Mbar_k = same with e^{-2 eps}.  Returns (M[0..3], Mbar[0..2]).
    Evaluated from A(beta) = C_D m^D x^{-(D-1)/2} K_nu(x) by exact differentiation (mpmath.diff).
    """
    with mp.workdps(dps):
        zeta = mp.mpf(zeta)
        nu = mp.mpf(D + 1) / 2
        CD = mp.gamma(mp.mpf(D) / 2) * mp.power(2, mp.mpf(D - 1) / 2) / mp.sqrt(mp.pi)

        def A(b):
            x = b * zeta
            return CD * zeta ** D * mp.power(x, -(mp.mpf(D) - 1) / 2) * mp.besselk(nu, x)

        M = [(-1) ** k * mp.diff(A, 1, k) for k in range(4)]
        Mb = [(-1) ** k * mp.diff(A, 2, k) for k in range(3)]
        return M, Mb


def c1_old_determinant(zeta, D=3, dps=50):
    """c1 from the OLD manuscript formula c1 = -Delta/(M0 M2 - M1^2)^2; returns (c1, Delta, G)."""
    with mp.workdps(dps):
        M, Mb = bg_moments_old_convention(zeta, D, dps)
        Delta = mp.det(mp.matrix([[M[2], M[1], M[0]], [M[3], M[2], M[1]], [Mb[2], Mb[1], Mb[0]]]))
        G = M[0] * M[2] - M[1] ** 2
        return -Delta / G ** 2, Delta, G


def R0_from_virial(A, B):
    r"""General dilute-limit curvature for phi = z A(beta) + z^2 B(beta) + O(z^3), z = e^{-gamma}.

    A, B: callables of beta (mpmath).  Returns the function beta -> R_0 (geometer's sign):
      R_0 = -[4(A''^2 - A'A''')B + 2(AA''' - A'A'')B' - (AA'' - A'^2)B''] / (2 (AA'' - A'^2)^2)
          =  B_2 + (L'''/L''^2) B_2' - B_2''/(2 L''),   L = ln A,  B_2 = -B/A^2.
    """
    def R0(beta):
        Ad = [mp.diff(A, beta, k) for k in range(4)]
        Bd = [mp.diff(B, beta, k) for k in range(3)]
        G = Ad[0] * Ad[2] - Ad[1] ** 2
        num = (4 * (Ad[2] ** 2 - Ad[1] * Ad[3]) * Bd[0] + 2 * (Ad[0] * Ad[3] - Ad[1] * Ad[2]) * Bd[1]
               - G * Bd[2])
        return -num / (2 * G ** 2)
    return R0


def Y_kappa(zeta, D=3, dps=50):
    r"""O(kappa^2) coefficient of the classical Kaniadakis curvature (geometer's sign):

        R_kappa^cl = -kappa^2 Y(zeta) / n + O(kappa^4),   n = z A(beta) (classical density),
        Y = (1/4) [2 c2 + c4/c2 - c3^2/c2^2]  =  beta^2 A H_3 / (4 H_2^2),
        H_2 = M0 M2 - M1^2, H_3 = det[M_{i+j}]_{i,j=0..2},  M_k = int rho eps^k e^{-beta eps}
        (equivalently R = -kappa^2 beta^2 H_3 / (4 z H_2^2) with phi = z A + ...)
    UR: Y -> (D+1)/2 ;  NR: Y -> (D+2)/4.
    """
    with mp.workdps(dps):
        _, c2, c3, c4 = mj_cumulants(zeta, D, dps)
        return (2 * c2 + c4 / c2 - c3 ** 2 / c2 ** 2) / 4


def A_reduced(zeta, D=3, kinetic=True, dps=50):
    """A = int p^{D-1} e^{-beta e} dp (kinetic, multiplies ztilde) or with eps (multiplies z); T=1."""
    with mp.workdps(dps):
        zeta = mp.mpf(zeta)
        nu = mp.mpf(D + 1) / 2
        CD = mp.gamma(mp.mpf(D) / 2) * mp.power(2, mp.mpf(D - 1) / 2) / mp.sqrt(mp.pi)
        if zeta == 0:
            return mp.gamma(D)
        val = CD * zeta ** D * mp.power(zeta, -(mp.mpf(D) - 1) / 2) * mp.besselk(nu, zeta)
        return val * mp.exp(zeta) if kinetic else val


def zstar_small_kappa(kappa, zeta, D=3, dps=50):
    r"""Leading small-kappa crossover of the Kaniadakis Bose curvature (activity ztilde):

        c1 = kappa^2 Y / n*  ->  ztilde* = kappa^2 Y(zeta) / (c1(zeta) A_kin(zeta)) + O(kappa^4 log^p)
    """
    with mp.workdps(dps):
        return mp.mpf(kappa) ** 2 * Y_kappa(zeta, D, dps) / (c1_closed(zeta, D, dps) * A_reduced(zeta, D, True, dps))


# =============================================================================
# 5. Non-relativistic / ultrarelativistic quantum gases in closed form
#    phi = c beta^{-s} f(gamma),  f = f_{s+1}(z) = -(1/a) Li_{s+1}(-a z)
# =============================================================================


def fermi_dirac_f(nu, z, a):
    """f_nu(z) = -(1/a) Li_nu(-a z)  (a=+1 Fermi, a=-1 Bose; a=0 -> z)."""
    if a == 0:
        return mp.mpf(z)
    return -mp.re(mp.polylog(nu, -a * mp.mpf(z))) / a     # real for 0 < z < 1 (mpmath may return mpc)


def R_powerlaw_quantum(z, a, s, beta=1, c=1, dps=60):
    r"""Exact curvature of phi = c beta^{-s} f_{s+1}(z) (power-law DOS, e.g. NR s=D/2, UR s=D):

      R = -(s+1) beta^s [2 f_{s+1} f_{s-1}^2 - f_{s+1} f_s f_{s-2} - f_s^2 f_{s-1}]
          / ( 2 c [(s+1) f_{s+1} f_{s-1} - s f_s^2]^2 ).
    D = 3 NR: s = 3/2, beta^s/c = lambda^3/g (phi = g lambda^{-3} f_{5/2}).
    D = 3 UR (massless, reduced units): s = 3, c = Gamma(3) = 2.
    The numerator is O(z^4) while each term is O(z^3): evaluated at `dps` digits.
    """
    with mp.workdps(dps):
        s = mp.mpf(s)
        f = lambda nu: fermi_dirac_f(nu, z, a)
        F1, F0, Fm1, Fm2 = f(s + 1), f(s), f(s - 1), f(s - 2)
        num = 2 * F1 * Fm1 ** 2 - F1 * F0 * Fm2 - F0 ** 2 * Fm1
        den = (s + 1) * F1 * Fm1 - s * F0 ** 2
        R = -(s + 1) * mp.mpf(beta) ** s * num / (2 * c * den ** 2)
    return +R


# =============================================================================
# 6. Root finding
# =============================================================================


def find_sign_changes(xs, Rs):
    """Indices i with sign(R_i) != sign(R_{i+1}) (ignoring exact zeros)."""
    out = []
    for i in range(len(xs) - 1):
        if Rs[i] == 0 or Rs[i + 1] == 0:
            continue
        if not (math.isfinite(Rs[i]) and math.isfinite(Rs[i + 1])):
            continue
        if (Rs[i] < 0) != (Rs[i + 1] < 0):
            out.append(i)
    return out


def zstar_root(zeta, st, lo_alpha, hi_alpha, D=3, mp_dps=None, xtol=1e-13):
    """Root of Rhat(alpha) = 0 between two alpha values (alpha = -ln ztilde).  Returns ztilde*.

    Uses the fast path (scipy brentq) or, with mp_dps, mpmath.findroot (anderson/bisection).
    """
    if mp_dps is None:
        from scipy.optimize import brentq
        f = lambda al: curvature_np(zeta, al, st, D=D)
        al = brentq(f, lo_alpha, hi_alpha, xtol=xtol, rtol=1e-14)
        return math.exp(-al), al
    with mp.workdps(mp_dps):
        f = lambda al: curvature_mp(zeta, al, st, D=D, dps=mp_dps)
        al = mp.findroot(f, (mp.mpf(lo_alpha), mp.mpf(hi_alpha)), solver="anderson")
        return mp.exp(-al), al
