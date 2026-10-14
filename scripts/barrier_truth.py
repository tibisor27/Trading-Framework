"""Adevarul de referinta pentru problema de bariera dubla.

Doua metode independente: formula inchisa si Monte Carlo direct.
Daca nu sunt de acord, una din ele e gresita -- si pana afli care,
nu ai voie sa crezi niciun numar iesit din sim_ftmo.py.
"""
import numpy as np


def closed_form(mu, sigma, up, down):
    """P(atingi +up inainte de -down), fara limita de timp.
    mu, sigma pe zi; up, down in aceleasi unitati (procente de cont)."""
    if mu == 0.0:
        return down / (up + down)          # ruina jucatorului
    theta = 2.0 * mu / sigma ** 2
    L, x = up + down, down
    return (1.0 - np.exp(-theta * x)) / (1.0 - np.exp(-theta * L))


def monte_carlo(mu, sigma, up, down, n_sims=100_000, max_days=4000,
                seed=0, chunk=2000):
    rng = np.random.default_rng(seed)
    wins, times, done = 0, [], 0
    while done < n_sims:
        m = min(chunk, n_sims - done)
        eq = np.cumsum(rng.normal(mu, sigma, size=(m, max_days)), axis=1)
        hu, hd = eq >= up, eq <= -down
        t_up = np.where(hu.any(1), hu.argmax(1), max_days)
        t_dn = np.where(hd.any(1), hd.argmax(1), max_days)
        wins += int((t_up < t_dn).sum())
        times.append(np.minimum(t_up, t_dn))
        done += m
    return wins / n_sims, float(np.concatenate(times).mean())


FAZE = [("faza 1", 10.0, 10.0), ("faza 2", 5.0, 10.0)]

print("=== 1. Fara edge: raspunsul NU depinde de sigma ===")
for sigma in (0.75, 1.50, 3.00):
    p1, _ = monte_carlo(0.0, sigma, 10.0, 10.0)
    p2, _ = monte_carlo(0.0, sigma, 5.0, 10.0)
    print(f"sigma {sigma:4.2f}   faza1 {p1:.3f} (0.500)   "
          f"faza2 {p2:.3f} (0.667)   ambele {p1*p2:.3f} (0.333)")

print("\n=== 2. Cu edge: formula vs Monte Carlo ===")
SHARPE = 0.95
for sigma in (0.25, 0.50, 0.75, 1.00, 1.50):
    mu = SHARPE * sigma / np.sqrt(252.0)
    linie = f"sigma {sigma:4.2f}"
    for nume, up, dn in FAZE:
        cf = closed_form(mu, sigma, up, dn)
        mc, zile = monte_carlo(mu, sigma, up, dn)
        linie += f"   {nume}: {cf:.4f} / {mc:.4f} ({zile:4.0f}z)"
    print(linie)