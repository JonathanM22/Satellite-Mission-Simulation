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

    def __init__(self, mass, epoch, r0=0, v0=0, celestial_body=None, label=None, color=None):
        self.mass = mass
        self.epoch = epoch
        self.r_ar = []
        self.v_ar = []
        self.energy_ar = []

        # Atr mostly for plotting
        self.label = label
        self.color = color

        if celestial_body != None:
            solar_system_ephemeris.set('de432s')
            self.label = celestial_body

            # Position of Sun @ epoch
            r_sun1, v_sun1 = get_body_barycentric_posvel('sun', epoch)

            r, v = get_body_barycentric_posvel(celestial_body, epoch)

            # Position & Velocity of earth respect to sun @ epoch
            self.r0 = (r.xyz - r_sun1.xyz).to(u.m).value   # type: ignore
            self.v0 = (v.xyz - v_sun1.xyz).to(u.m/u.s).value  # type: ignore
        if celestial_body == None:
            self.r0 = r0
            self.v0 = v0
