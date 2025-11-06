import numpy as np
import math

def stumpff_C2(psi):
    if psi > 1e-6:
        return (1 - np.cos(np.sqrt(psi))) / psi
    elif psi < -1e-6:
        return (np.cosh(np.sqrt(-psi)) - 1) / (-psi)
    else:
        return 1/2 - psi/24 + psi**2/720

def stumpff_C3(psi):
    if psi > 1e-6:
        return (np.sqrt(psi) - np.sin(np.sqrt(psi))) / (psi * np.sqrt(psi))
    elif psi < -1e-6:
        return (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / ((-psi) * np.sqrt(-psi))
    else:
        return 1/6 - psi/120 + psi**2/5040

def universal_lambert(r1_vec, r2_vec, TOF, mu=1.0, tm=1):
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    cos_dnu = np.dot(r1_vec, r2_vec) / (r1 * r2)
    A = tm * np.sqrt(r1 * r2 * (1 + cos_dnu))
    if A == 0:
        raise ValueError("Transfer angle is 180°, Lambert undefined")

    psi = 0.0
    psi_low = -4 * math.pi**2
    psi_up = 4 * math.pi**2
    eps = 1e-8
    nmax = 1000
    c2 = stumpff_C2(psi)
    c3 = stumpff_C3(psi)
    y = r1 + r2 + A * (psi * c3 - 1) / np.sqrt(c2)
    chi = np.sqrt(y / c2)
    dt = (chi**3 * c3 + A * np.sqrt(y)) / np.sqrt(mu)

    # Newton iteration on psi
    for _ in range(nmax):
        c2 = stumpff_C2(psi)
        c3 = stumpff_C3(psi)
        y = r1 + r2 + A * (psi * c3 - 1) / np.sqrt(c2)
        if y < 0:
            psi = (psi + psi_low) / 2
            continue
        chi = np.sqrt(y / c2)
        dt = (chi**3 * c3 + A * np.sqrt(y)) / np.sqrt(mu)
        if abs(dt - TOF) < eps:
            break
        if dt <= TOF:
            psi_low = psi
        else:
            psi_up = psi
        psi = (psi_up + psi_low) / 2

    f = 1 - y / r1
    g = A * np.sqrt(y / mu)
    gdot = 1 - y / r2

    v1_vec = (r2_vec - f * r1_vec) / g
    v2_vec = (gdot * r2_vec - r1_vec) / g

    return v1_vec, v2_vec

# Example
v1, v2 = universal_lambert(
    np.array([1.01566, 0, 0]),
    np.array([0.387926, 0.183961, 0.551884]),
    TOF=5,
    mu=1.0,
    tm=1
)

print("v1_vec =", v1)
print("v2_vec =", v2)