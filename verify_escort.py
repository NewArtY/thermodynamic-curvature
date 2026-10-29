# -*- coding: utf-8 -*-
r"""
verify_escort.py -- deep-dilute curvature of the escort-weighted quantum Tsallis gas (Sec. 5.1).

The thermodynamically consistent prescription of Cleymans and Worku uses the density weight
n = f^q, where f = 1/(u + a), u = [1 + (q-1) y]^{1/(q-1)}, is the occupation of Eq. (3).
For a = 0 this is the flat classical escort gas (Corollary 1c); for a = +-1 it differs from
Eq. (3) and from {[1+(q-1)y]^{q/(q-1)} + a}^{-1}.

Independent check (no thermogeom.py): the derivatives of phi are evaluated by mpmath
quadrature from n, n', n'' in closed form, and R from Eq. (8) of the article.
Claims checked (tau = 1/(q-1), gbar = gamma + tau, beta = 1, Rhat = (g T^3/2 pi^2) R):
  bare occupation (3):   Rhat * gbar^4 -> -a C(q),   C   = (t-1)^2 (t-2) (t-3)^2 (t-4) / [4 (2t-1)(2t-3)]
  escort  n = f^q:       Rhat * gbar^3 -> -a C_e(q), C_e = t (t-1)^2 (t-2)^2 (t-3) / [4 (t+1)(2t-1)(2t-3)]
Outputs data/escort_check.json
"""
import json
import os

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "escort_check.json")


def C_bare(t):
    return (t - 1) ** 2 * (t - 2) * (t - 3) ** 2 * (t - 4) / (4 * (2 * t - 1) * (2 * t - 3))


def C_escort(t):
    return t * (t - 1) ** 2 * (t - 2) ** 2 * (t - 3) / (4 * (t + 1) * (2 * t - 1) * (2 * t - 3))


def n_derivs(y, q, a, escort):
    """Return (n', n'') with respect to y."""
    tau = 1 / (q - 1)
    t = 1 + (q - 1) * y
    u = t ** tau
    du = t ** (tau - 1)                       # d u / d y, since tau (q - 1) = 1
    d2u = (tau - 1) * (q - 1) * t ** (tau - 2)
    s = q if escort else mp.mpf(1)            # n = (u + a)^(-s)
    w = u + a
    n1 = -s * w ** (-s - 1) * du
    n2 = s * (s + 1) * w ** (-s - 2) * du ** 2 - s * w ** (-s - 1) * d2u
    return n1, n2


def Rhat(zeta, gamma, q, a, escort, dps=80):
    """Reduced curvature at beta = 1, mass zeta, prefactor g/(2 pi^2) = 1."""
    with mp.workdps(dps):
        q, a, zeta, gamma = mp.mpf(q), mp.mpf(a), mp.mpf(zeta), mp.mpf(gamma)

        def mom(i, order):
            def f(p):
                eps = mp.sqrt(p * p + zeta * zeta)
                n1, n2 = n_derivs(eps + gamma, q, a, escort)
                return p * p * eps ** i * (-(n1 if order == 2 else n2))
            gb = gamma + 1 / (q - 1)
            return mp.quad(f, [0, 1, gb, 10 * gb, mp.inf])

        g = {k: mom(k, 2) for k in (0, 1, 2)}          # phi_gg, phi_bg, phi_bb
        h = {k: mom(k, 3) for k in (0, 1, 2, 3)}       # third derivatives
        M = mp.matrix([[g[2], g[1], g[0]],
                       [h[3], h[2], h[1]],
                       [h[2], h[1], h[0]]])
        detg = g[2] * g[0] - g[1] ** 2
        return -mp.det(M) / (2 * detg ** 2)


def main():
    rows = []
    for q in (mp.mpf("1.1"), mp.mpf("1.2")):
        tau = 1 / (q - 1)
        for a in (1, -1):
            for zeta in (0, 1):
                for gamma in (mp.mpf(100), mp.mpf(1000), mp.mpf(10000)):
                    gb = gamma + tau
                    rb = Rhat(zeta, gamma, q, a, escort=False)
                    re = Rhat(zeta, gamma, q, a, escort=True)
                    row = dict(q=float(q), a=a, zeta=zeta, gamma=float(gamma),
                               bare_ratio=float(rb * gb ** 4 / (-a * C_bare(tau))),
                               escort_ratio=float(re * gb ** 3 / (-a * C_escort(tau))))
                    rows.append(row)
                    print("q=%.1f a=%+d zeta=%d gamma=%-6g  bare Rhat*gbar^4/(-aC)=%.10f"
                          "  escort Rhat*gbar^3/(-aC_e)=%.10f"
                          % (q, a, zeta, gamma, row["bare_ratio"], row["escort_ratio"]))
    plateau = {}
    for qs in ("1.01", "1.1", "1.2"):
        tau = 1 / (mp.mpf(qs) - 1)
        plateau[qs] = dict(bare=float(C_bare(tau) / tau ** 4), escort=float(C_escort(tau) / tau ** 3),
                           bg=1 / 16)
        print("q=%s  massless plateau: bare %.5f  escort %.5f  (BG 1/16 = 0.0625; ratios %.2f, %.2f)"
              % (qs, plateau[qs]["bare"], plateau[qs]["escort"],
                 0.0625 / plateau[qs]["bare"], 0.0625 / plateau[qs]["escort"]))
    with open(OUT, "w") as fh:
        json.dump(dict(rows=rows, plateau=plateau,
                       C_e={"1.1": float(C_escort(mp.mpf(10))), "1.2": float(C_escort(mp.mpf(5)))}),
                  fh, indent=1)
    print("wrote data/escort_check.json")


if __name__ == "__main__":
    main()
