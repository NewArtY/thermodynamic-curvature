# -*- coding: utf-8 -*-
r"""
verify_convention.py -- which curvature does the "determinant formula" compute?
==============================================================================

For a 2D Hessian metric g_ij = d_i d_j phi the manuscript uses

    R_det = - det[[phi_11, phi_12, phi_22],
                  [phi_111, phi_112, phi_122],
                  [phi_112, phi_122, phi_222]] / (2 (det g)^2).

We compute, completely independently, the Ricci scalar from Christoffel symbols
in the GEOMETER's convention

    R^r_{s m n} = d_m Gamma^r_{n s} - d_n Gamma^r_{m s}
                  + Gamma^r_{m l} Gamma^l_{n s} - Gamma^r_{n l} Gamma^l_{m s},
    Ric_{s n}   = R^r_{s r n},       R = g^{s n} Ric_{s n},

for which the unit 2-sphere has R = +2 (Gaussian curvature K = R/2 = +1), and
also via the Brioschi formula for the Gaussian curvature K.  Tests:

  T1  unit sphere                         -> R = +2
  T2  Poincare half plane (dx^2+dy^2)/y^2 -> R = -2
  T3  GENERIC Hessian metric (phi = arbitrary sympy Function): R_Christoffel - R_det == 0
  T4  Fisher metric of the Gaussian family (Hessian of its log-partition psi):
      known constant curvature K = -1/2  -> R = -1; check R_det = -1
  T5  the actual NR ideal Bose / Fermi gas potential phi = beta^{-3/2} f(gamma) with
      f the polylog: R_Christoffel vs R_det at sample points (mpmath evaluation)

Result is printed and saved to data/convention_check.json.
"""
import json
import os

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)


def ricci_scalar_christoffel(g, coords):
    """Ricci scalar (geometer's convention: sphere > 0) of a metric matrix g."""
    n = len(coords)
    ginv = sp.simplify(g.inv())
    Gam = [[[sum(ginv[r, l] * (sp.diff(g[l, m], coords[s]) + sp.diff(g[l, s], coords[m])
                                - sp.diff(g[m, s], coords[l])) for l in range(n)) / 2
             for s in range(n)] for m in range(n)] for r in range(n)]
    # Gam[r][m][s] = Gamma^r_{m s}

    def Riem(r, s, m, nn):
        val = sp.diff(Gam[r][nn][s], coords[m]) - sp.diff(Gam[r][m][s], coords[nn])
        val += sum(Gam[r][m][l] * Gam[l][nn][s] - Gam[r][nn][l] * Gam[l][m][s] for l in range(n))
        return val

    Ric = sp.zeros(n, n)
    for s in range(n):
        for nn in range(n):
            Ric[s, nn] = sum(Riem(r, s, r, nn) for r in range(n))
    return sum(ginv[s, nn] * Ric[s, nn] for s in range(n) for nn in range(n))


def gauss_brioschi(E, F, G, u, v):
    """Gaussian curvature K via the Brioschi formula (geometer's sign, sphere K=+1)."""
    Eu, Ev = sp.diff(E, u), sp.diff(E, v)
    Fu, Fv = sp.diff(F, u), sp.diff(F, v)
    Gu, Gv = sp.diff(G, u), sp.diff(G, v)
    M1 = sp.Matrix([[-sp.diff(E, v, 2) / 2 + sp.diff(F, u, v) - sp.diff(G, u, 2) / 2, Eu / 2, Fu - Ev / 2],
                    [Fv - Gu / 2, E, F],
                    [Gv / 2, F, G]])
    M2 = sp.Matrix([[0, Ev / 2, Gu / 2],
                    [Ev / 2, E, F],
                    [Gu / 2, F, G]])
    return (M1.det() - M2.det()) / (E * G - F ** 2) ** 2


def R_det_formula(phi, x, y):
    """The manuscript's determinant curvature for the Hessian metric of phi(x, y)."""
    d = lambda *a: sp.diff(phi, *a)
    gxx, gxy, gyy = d(x, x), d(x, y), d(y, y)
    M = sp.Matrix([[gxx, gxy, gyy],
                   [d(x, x, x), d(x, x, y), d(x, y, y)],
                   [d(x, x, y), d(x, y, y), d(y, y, y)]])
    detg = gxx * gyy - gxy ** 2
    return -M.det() / (2 * detg ** 2)


def main():
    out = {}
    th, ph = sp.symbols("theta phi", positive=True)
    # T1: unit sphere
    g = sp.Matrix([[1, 0], [0, sp.sin(th) ** 2]])
    R1 = sp.simplify(ricci_scalar_christoffel(g, [th, ph]))
    K1 = sp.simplify(gauss_brioschi(1, 0, sp.sin(th) ** 2, th, ph))
    print("T1 unit sphere:            R_Christoffel =", R1, "  K_Brioschi =", K1)
    out["T1_sphere"] = {"R": str(R1), "K": str(K1)}

    # T2: Poincare half-plane
    xx, yy = sp.symbols("x y", positive=True)
    g = sp.Matrix([[1 / yy ** 2, 0], [0, 1 / yy ** 2]])
    R2 = sp.simplify(ricci_scalar_christoffel(g, [xx, yy]))
    print("T2 hyperbolic half-plane:  R_Christoffel =", R2)
    out["T2_hyperbolic"] = {"R": str(R2)}

    # T3: generic Hessian metric
    b, c = sp.symbols("beta gamma", real=True)
    f = sp.Function("phi")(b, c)
    g = sp.Matrix([[sp.diff(f, b, b), sp.diff(f, b, c)], [sp.diff(f, b, c), sp.diff(f, c, c)]])
    Rch = ricci_scalar_christoffel(g, [b, c])
    Rdet = R_det_formula(f, b, c)
    Kbr = gauss_brioschi(g[0, 0], g[0, 1], g[1, 1], b, c)
    diff1 = sp.simplify(sp.together(Rch - Rdet))
    diff2 = sp.simplify(sp.together(2 * Kbr - Rdet))
    print("T3 generic Hessian metric: simplify(R_Christoffel - R_det) =", diff1,
          ";  simplify(2K_Brioschi - R_det) =", diff2)
    out["T3_generic_hessian"] = {"R_christoffel_minus_R_det": str(diff1),
                                 "2K_brioschi_minus_R_det": str(diff2)}

    # T4: Gaussian family, psi(t1,t2) = -t1^2/(4 t2) + 1/2 log(-pi/t2), t2<0
    t1 = sp.symbols("t1", real=True)
    t2 = sp.symbols("t2", negative=True)
    psi = -t1 ** 2 / (4 * t2) + sp.Rational(1, 2) * sp.log(-sp.pi / t2)
    g = sp.hessian(psi, (t1, t2))
    R4c = sp.simplify(ricci_scalar_christoffel(g, [t1, t2]))
    R4d = sp.simplify(R_det_formula(psi, t1, t2))
    print("T4 Gaussian-family Fisher metric (Hessian of log-partition): R_Christoffel =", R4c,
          "  R_det =", R4d, "  (known: K=-1/2 -> R=-1)")
    out["T4_gaussian_family"] = {"R_christoffel": str(R4c), "R_det": str(R4d)}

    # T5: NR quantum gases phi = beta^{-s} f(gamma), f = -(1/a) Li_{s+1}(-a e^{-gamma})
    s = sp.Rational(3, 2)
    res5 = []
    for a in (+1, -1):
        fexpr = -sp.Rational(1, a) * sp.polylog(s + 1, -a * sp.exp(-c))
        phi = b ** (-s) * fexpr
        g = sp.hessian(phi, (b, c))
        Rc = ricci_scalar_christoffel(g, [b, c])
        Rd = R_det_formula(phi, b, c)
        for (bv, cv) in ((1.0, 0.5), (0.7, 2.0), (2.0, 0.05)):
            rc = complex(sp.N(Rc.subs({b: bv, c: cv}), 30))
            rd = complex(sp.N(Rd.subs({b: bv, c: cv}), 30))
            print("T5 NR %s gas  beta=%.2f gamma=%.2f:  R_Christoffel=%.15e  R_det=%.15e"
                  % ("Fermi" if a == 1 else "Bose ", bv, cv, rc.real, rd.real))
            res5.append({"a": a, "beta": bv, "gamma": cv, "R_christoffel": rc.real, "R_det": rd.real})
    out["T5_NR_quantum_gas"] = res5
    with open(os.path.join(DATA, "convention_check.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
