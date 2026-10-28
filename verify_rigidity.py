# -*- coding: utf-8 -*-
r"""
verify_rigidity.py -- infinitesimal converse of the flatness criterion (N1)
==========================================================================

Deform the Boltzmann single-mode potential, F(y) = e^{-y} [1 + eps h(y)], with h a
polynomial, for a GENERIC one-particle function A(beta) = int rho e^{-beta eps} (arbitrary
density of states).  Then

    phi = e^{-gamma} [ A(beta) + eps Q_h(beta, gamma) ],
    Q_{y^k} = sum_j C(k,j) gamma^{k-j} (-beta)^j A^{(j)}(beta).

The curvature is O(eps): R = eps * dR[h] + O(eps^2).  We compute dR[y^k] for k = 0..6
symbolically and test which vanish identically (for arbitrary A).  Expected:
k = 0, 1, 2 (normalisation, affine rescaling y -> (1+c)y, first-order Tsallis) give 0;
k >= 3 give non-zero curvature.  Also: the first-order Kaniadakis deformation
h = (y^3 + 3y^2 + 6y + 6)/6 reproduces -beta^2 H3/(4 z H2^2).
Output data/rigidity.json
"""
import json
import os

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
b, g, eps = sp.symbols("beta gamma epsilon")
A = sp.Function("A")(b)


def Q_power(k):
    return sum(sp.binomial(k, j) * g ** (k - j) * (-b) ** j * sp.diff(A, b, j) for j in range(k + 1))


def first_order_R(Q):
    phi = sp.exp(-g) * (A + eps * Q)
    d = lambda *a: sp.diff(phi, *a)
    M = sp.Matrix([[d(b, b), d(b, g), d(g, g)],
                   [d(b, b, b), d(b, b, g), d(b, g, g)],
                   [d(b, b, g), d(b, g, g), d(g, g, g)]])
    D = M.det()
    G = (d(b, b) * d(g, g) - d(b, g) ** 2).subs(eps, 0)
    D1 = sp.diff(D, eps).subs(eps, 0)
    return sp.simplify(sp.expand(-D1 / (2 * G ** 2)))


def main():
    out = {}
    for k in range(0, 7):
        Rk = first_order_R(Q_power(k))
        zero = (Rk == 0)
        out["y^%d" % k] = str(sp.factor(Rk)) if not zero else "0"
        print("h = y^%d :  dR = %s" % (k, "0" if zero else sp.factor(Rk)))
    # independent of gamma?  (k=3)
    R3 = first_order_R(Q_power(3))
    print("k=3: d/dgamma (dR e^{-gamma}) =", sp.simplify(sp.diff(R3 * sp.exp(-g), g)))
    with open(os.path.join(HERE, "data", "rigidity.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
