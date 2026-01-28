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

    def r_at_true_anomaly(self, f, deg=False):
        if deg == True:
            f = np.deg2rad(f)
        return self.p / (1 + self.e*np.cos(f))

    @property
    def energy(self):
        return (-self.mu) / (2*self.a)

    @property
    def p(self):
        return self.a * (1 - self.e**2)

    @property
    def period(self):
        return (2*np.pi)*np.sqrt(self.a**3/self.mu)
