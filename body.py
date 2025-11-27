"""
Body class 
"""
import numpy as np
from scipy.integrate import ode
from scipy import optimize

import astropy
from astropy.time import Time
from astropy.time import TimeDelta
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel


class Body:

    def __init__(self, mass, epoch, r0=0, v0=0, celestial_body=False, label='unlabeled', color='black'):
        self.mass = mass
        self.epoch = epoch
        self.r_ar = []  # position array
        self.v_ar = []  # velocity array
        self.t_ar = []  # time array

        # Atr mostly for plotting
        self.label = label
        self.color = color

        if celestial_body != False:
            solar_system_ephemeris.set('de432s')
            self.label = celestial_body
            r, v = get_body_barycentric_posvel(
                celestial_body, epoch)

            self.r0 = r.xyz.to(u.km)
            self.v0 = v.xyz.to(u.km/u.s)
        else:
            self.r0 = r0
            self.v0 = v0


class Spacecraft(Body):

    def __init__(self, mass, epoch, r0=0, v0=0, celestial_body=False, label=None, color=None):
        super().__init__(mass, epoch, r0=0, v0=0,
                         celestial_body=False, label=None, color=None)

        self.inertia = np.zeros((3, 3))
        self.energy_ar = []
        self.h_ar = []
        self.central_body = None
        self.model = ''
