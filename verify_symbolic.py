# -*- coding: utf-8 -*-
r"""
verify_symbolic.py -- sympy proofs of the analytic statements N1, N2, N4, N5
=============================================================================

S1 (N1) affine-homothety identity: for X = (al*b + b1) d_b + (al*g + b2) d_g and ANY phi,
        d_i d_j (X phi) = X(phi_ij) + 2 al phi_ij.
     Hence X phi = c phi + affine  =>  X^b row2 + X^g row3 = (c - 2 al) row1  =>  det = 0, R = 0.
S2 (N1) homogeneity of the classical Tsallis single-mode potential (bare and escort):
        (y + 1/(q-1)) F'(y) = d F(y),  d = (q-2)/(q-1)  [bare],  d = -1/(q-1)  [escort].
S3 (N1, converse) separable phi = cosh(b) + exp(g): R = 0 identically, yet no affine
     homothety exists.  Two-level ideal gas: flat for ANY statistics F.
S4 (N2) dilute limit of R for phi = z A + z^2 B + z^3 C (generic functions):
        R_0 = -[4(A''^2 - A'A''')B + 2(AA''' - A'A'')B' - (AA'' - A'^2)B''] / (2 (AA'' - A'^2)^2)
     and its virial form  R_0 = B2 + (L'''/L''^2) B2' - B2''/(2 L''),  L = ln A, B2 = -B/A^2.
S5 (N4) O(kappa^2) classical Kaniadakis curvature for a GENERIC one-particle function A(beta):
        R = -kappa^2 beta^2 H3 / (4 z H2^2) + O(kappa^4),   gamma-independent,
     H2 = M0 M2 - M1^2, H3 = Hankel det of M_k = (-1)^k A^(k).
S6 (N5) deep-dilute constants for massless D=3 power-law potentials
        phi = beta^{-3}[C1 G^{e1} + eps C2 G^{e2}]:  R = eps * (...) * beta^3 G^{e2 - 2 e1} + O(eps^2).
Writes data/symbolic_checks.json.
"""
import json
import os

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "symbolic_checks.json")

b, g = sp.symbols("beta gamma", real=True)


def det_rows(phi, x, y):
    d = lambda *a: sp.diff(phi, *a)
    r1 = [d(x, x), d(x, y), d(y, y)]
    r2 = [d(x, x, x), d(x, x, y), d(x, y, y)]
    r3 = [d(x, x, y), d(x, y, y), d(y, y, y)]
    return r1, r2, r3


def Rdet(phi, x, y):
    r1, r2, r3 = det_rows(phi, x, y)
    M = sp.Matrix([r1, r2, r3])
    G = r1[0] * r1[2] - r1[1] ** 2
    return -M.det() / (2 * G ** 2)


def main():
    out = {}
    # ---------------- S1
    al, b1, b2, c = sp.symbols("alpha_l b1 b2 c")
    phi = sp.Function("phi")(b, g)
    X = lambda f: (al * b + b1) * sp.diff(f, b) + (al * g + b2) * sp.diff(f, g)
    ok = True
    for (i, j) in ((b, b), (b, g), (g, g)):
        lhs = sp.diff(X(phi), i, j)
        rhs = X(sp.diff(phi, i, j)) + 2 * al * sp.diff(phi, i, j)
        ok = ok and sp.simplify(lhs - rhs) == 0
    print("S1 d_i d_j (X phi) = X(phi_ij) + 2 alpha phi_ij  for generic phi:", ok)
    out["S1_affine_identity"] = bool(ok)

    # ---------------- S2
    q, y = sp.symbols("q y", positive=True)
    Fbare = (1 + (q - 1) * y) ** ((q - 2) / (q - 1)) / (2 - q)
    Fesc = (1 + (q - 1) * y) ** (-1 / (q - 1))
    s_bare = sp.simplify((y + 1 / (q - 1)) * sp.diff(Fbare, y) - (q - 2) / (q - 1) * Fbare)
    s_esc = sp.simplify((y + 1 / (q - 1)) * sp.diff(Fesc, y) + 1 / (q - 1) * Fesc)
    n_bare = sp.simplify(-sp.diff(Fbare, y))
    n_esc = sp.simplify(-sp.diff(Fesc, y))
    print("S2 bare Tsallis: n =", n_bare, "; Euler residual:", s_bare)
    print("S2 escort Tsallis: n =", n_esc, "; Euler residual:", s_esc)
    out["S2"] = {"bare_residual": str(s_bare), "escort_residual": str(s_esc),
                 "n_bare": str(n_bare), "n_escort": str(n_esc)}

    # ---------------- S3: converse fails
    phis = sp.cosh(b) + sp.exp(g)
    Rs = sp.simplify(Rdet(phis, b, g))
    lam, bb1, bb2, cc, l0, l1, l2 = sp.symbols("lam bb1 bb2 cc l0 l1 l2")
    expr = sp.expand((lam * b + bb1) * sp.diff(phis, b) + (lam * g + bb2) * sp.diff(phis, g)
                     - cc * phis - l0 - l1 * b - l2 * g)
    # coefficients w.r.t. the independent functions {b sinh b, sinh b, cosh b, g e^g, e^g, 1, b, g}
    expr = expr.rewrite(sp.exp)
    eqs = sp.Poly(sp.expand(expr.subs({sp.exp(b): sp.Symbol("Eb"), sp.exp(-b): 1 / sp.Symbol("Eb"),
                                       sp.exp(g): sp.Symbol("Eg")}) * sp.Symbol("Eb")),
                  sp.Symbol("Eb"), sp.Symbol("Eg"), b, g).coeffs()
    sol = sp.solve(eqs, [lam, bb1, bb2, cc, l0, l1, l2], dict=True)
    print("S3 separable phi = cosh(beta)+exp(gamma): R =", Rs, "; affine homotheties:", sol)
    e1, e2 = sp.symbols("epsilon1 epsilon2", positive=True)
    Fgen = sp.Function("F")
    phi2 = Fgen(b * e1 + g) + Fgen(b * e2 + g)
    R2 = sp.simplify(Rdet(phi2, b, g))
    print("S3 two-level ideal gas with arbitrary F: R =", R2)
    out["S3"] = {"R_separable": str(Rs), "homotheties": str(sol), "R_two_level": str(R2)}

    # ---------------- S4: generic dilute expansion
    z = sp.symbols("z", positive=True)
    A, B, C = [sp.Function(nm)(b) for nm in ("A", "B", "C")]
    phi = sp.exp(-g) * A + sp.exp(-2 * g) * B + sp.exp(-3 * g) * C
    r1, r2, r3 = det_rows(phi, b, g)
    sub = lambda e: sp.expand(e.subs(sp.exp(-g), z).subs(sp.exp(-2 * g), z ** 2).subs(sp.exp(-3 * g), z ** 3))
    r1, r2, r3 = [[sub(e) for e in r] for r in (r1, r2, r3)]
    # exp(-k g) may appear as exp(-g)**k after differentiation; normalise:
    r1, r2, r3 = [[sp.expand(sp.powsimp(e).subs(sp.exp(g), 1 / z)) for e in r] for r in (r1, r2, r3)]
    Dm = sp.expand(sp.Matrix([r1, r2, r3]).det())
    G = sp.expand(r1[0] * r1[2] - r1[1] ** 2)
    D4 = Dm.coeff(z, 4)
    D3 = Dm.coeff(z, 3)
    G2 = G.coeff(z, 2)
    R0 = sp.simplify(-D4 / (2 * G2 ** 2))
    Ad = [sp.diff(A, b, k) for k in range(4)]
    Bd = [sp.diff(B, b, k) for k in range(3)]
    Gs = Ad[0] * Ad[2] - Ad[1] ** 2
    claim = -(4 * (Ad[2] ** 2 - Ad[1] * Ad[3]) * Bd[0] + 2 * (Ad[0] * Ad[3] - Ad[1] * Ad[2]) * Bd[1]
              - Gs * Bd[2]) / (2 * Gs ** 2)
    d1 = sp.simplify(R0 - claim)
    print("S4 coefficient z^3 of det (classical part) =", sp.simplify(D3),
          "; z^0 of det, z^1:", Dm.coeff(z, 0), Dm.coeff(z, 1), "; R0 - claim =", d1)
    # virial form
    B2 = sp.Function("B2")(b)
    L = sp.log(A)
    vir = B2 + sp.diff(L, b, 3) / sp.diff(L, b, 2) ** 2 * sp.diff(B2, b) - sp.diff(B2, b, 2) / (2 * sp.diff(L, b, 2))
    d2 = sp.simplify(claim.subs(B, -B2 * A ** 2).doit() - vir)
    print("S4 virial form: claim(B=-B2 A^2) - [B2 + L'''/L''^2 B2' - B2''/(2L'')] =", d2)
    out["S4"] = {"D3": str(sp.simplify(D3)), "R0_minus_claim": str(d1), "virial_form_residual": str(d2)}

    # ---------------- S5: O(kappa^2) Kaniadakis, generic A(beta)
    kap = sp.symbols("kappa", positive=True)
    # Q = (1/6) int rho e^{-beta eps} P(beta eps + gamma),  P(y) = y^3 + 3y^2 + 6y + 6.
    # int rho e^{-beta eps} (beta eps)^j = (-beta)^j A^(j)(beta)
    Mj = lambda j: (-b) ** j * sp.diff(A, b, j)
    Q = sp.Rational(1, 6) * (Mj(3) + 3 * g * Mj(2) + 3 * g ** 2 * Mj(1) + g ** 3 * Mj(0)
                             + 3 * (Mj(2) + 2 * g * Mj(1) + g ** 2 * Mj(0))
                             + 6 * (Mj(1) + g * Mj(0)) + 6 * Mj(0))
    phi = sp.exp(-g) * (A + kap ** 2 * Q)
    r1, r2, r3 = det_rows(phi, b, g)
    Dm = sp.Matrix([r1, r2, r3]).det()
    G = r1[0] * r1[2] - r1[1] ** 2
    Dk = sp.expand(sp.diff(Dm, kap, 2).subs(kap, 0) / 2)       # kappa^2 coefficient
    D0 = sp.simplify(Dm.subs(kap, 0))
    G0 = G.subs(kap, 0)
    Rk = sp.simplify(-Dk / (2 * G0 ** 2))
    M = [(-1) ** k * sp.diff(A, b, k) for k in range(5)]
    H2 = M[0] * M[2] - M[1] ** 2
    H3 = sp.Matrix([[M[0], M[1], M[2]], [M[1], M[2], M[3]], [M[2], M[3], M[4]]]).det()
    claimk = -b ** 2 * H3 / (4 * sp.exp(-g) * H2 ** 2)
    d3 = sp.simplify(Rk - claimk)
    print("S5 Kaniadakis: kappa^0 det =", D0, "; R_kappa2 - [-beta^2 H3/(4 z H2^2)] =", d3,
          "; d/dgamma (R_kappa2 * z) =", sp.simplify(sp.diff(Rk * sp.exp(-g), g)))
    out["S5"] = {"D0": str(D0), "residual": str(d3)}

    # ---------------- S6: massless power-law potentials
    C1, C2, e1s, e2s, eps, G_ = sp.symbols("C1 C2 e1 e2 epsilon Gp", positive=True)
    s = 3
    f = C1 * G_ ** e1s + eps * C2 * G_ ** e2s
    phi = b ** (-s) * f
    Rm = Rdet(phi, b, G_)
    R1 = sp.simplify(sp.diff(Rm, eps).subs(eps, 0))
    R0m = sp.simplify(Rm.subs(eps, 0))
    pref = sp.simplify(R1 / (b ** 3 * C2 / C1 ** 2 * G_ ** (e2s - 2 * e1s)))
    print("S6 pure power (eps=0): R =", R0m)
    print("S6 first order: R1 = (C2/C1^2) beta^3 G^(e2-2e1) * K(e1,e2),  K =", sp.factor(pref))
    out["S6"] = {"R_pure_power": str(R0m), "K_e1_e2": str(sp.factor(pref))}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
