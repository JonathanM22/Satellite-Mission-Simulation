"""
Orbit class 
"""
import numpy as np
from scipy.integrate import ode
from scipy import optimize


class Orbit:
    """
    Intializes orbit class with 6 orbital elements. 

    e: vector
    e_mag: magnitude 
    """

    def __init__(self, mu, a=None, e=None, f0=None, inc=None, raan=None, aop=None, e_vec=np.zeros(3)):
        self.a = a
        self.e = e
        self.f0 = f0
        self.e_vec = e_vec
        self.inc = inc
        self.raan = raan
        self.aop = aop
        self.mu = mu
        self.h = np.zeros(2)
        self.p = None
        self.energy = None

    def r_at_true_anomaly(self, f):
        """
        Returns r scalor using orbit equation 
        """
        try:
            return self.calc_p() / (1 + self.e*np.cos(f))
        except:
            raise ValueError(
                "r_at_true_anomaly cannot be calculated. Check a, e or f")

    def calc_energy(self):
        """
        Returns orbit energy 
        """
        if self.energy is not None:
            return self.energy
        else:
            try:
                return (-self.mu) / (2*self.a)  # type:ignore
            except:
                raise ValueError(
                    "Orbit energy  cannot be calculated. Check a")

    def calc_p(self):
        """
        Returns semilatus rectum  
        """
        if self.p is not None:
            return self.p
        else:
            try:
                return self.a * (1 - self.e**2)  # type:ignore
            except:
                raise ValueError(
                    "Orbit semilatus rectum cannot be calculated. Check a or e")

    def period(self):
        """
        Calculates period or orbit.
        Returns period in seconds
        """
        try:
            return (2*np.pi)*np.sqrt(self.a**3/self.mu)  # type:ignore
        except:
            raise ValueError(
                "Orbit has period cannot be calculated. Check a or mu")
