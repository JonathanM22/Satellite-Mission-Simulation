"""
Orbit class 
"""
import numpy as np
from scipy.integrate import ode
from scipy import optimize


class Orbit:
    """
    Intializes orbit class with 6 orbital elements. 
    """

    def __init__(self, mu, a=0, e=0, f0=0, inc=0, raan=0, aop=0, e_vec=np.zeros(3)):
        self.a = a
        self.e = e
        self.e_vec = e_vec
        self.f0 = f0
        self.inc = inc
        self.raan = raan
        self.aop = aop
        self.mu = mu
        self.h = 0
        self.p = 0
        self.energy = 0

    def r_at_true_anomaly(self, e, p, f):
        return p / (1 + e*np.cos(f))

    def calc_energy(self, a, mu):
        return (-mu) / (2*a)

    def calc_p(self, a, e):
        return a * (1 - e**2)

    def period(self, a, mu):
        return (2*np.pi)*np.sqrt(a**3/mu)
