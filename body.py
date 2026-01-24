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
from astropy import constants as const


class Body:

    def __init__(self, mass, epoch, r0=0, v0=0, celestial_body=False, label='unlabeled', color='black'):
        self.mass = mass
        self.epoch = epoch
        self.r_ar = np.zeros(3)  # position array
        self.v_ar = np.zeros(3)  # velocity array
        self.t_ar = np.zeros(3)  # time array
        self.energy_ar = np.zeros(1)
        self.h_ar = np.zeros(1)
        self.mu = 0

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
        self.model = ""
        self.w_ar = np.zeros(1)


MOON_MASS = (7.34 * 10**22) * u.kg
SAT_MASS = 100 * u.kg
G = const.G.to(u.km**3 / (u.kg * u.s**2))  # convert to km
SUN_MASS = const.M_sun
SUN_MU = const.GM_sun.to(u.km**3 / u.s**2).value
EARTH_MASS = const.M_earth
EARTH_MU = const.GM_earth.to(u.km**3 / u.s**2)
MARS_MASS = (6.39 * 10**23) * u.kg
MARS_RAD = 3390 * u.km
MARS_MU = MARS_MASS * G
JUPITER_MASS = const.M_jup
JUPITER_MU = const.GM_jup.to(u.km**3 / u.s**2)
SATURN_MASS = (5.638 * 10**26) * u.kg
URANAS_MASS = (8.681 * 10**25) * u.kg
NEPTUNE_MASS = (1.024 * 10**26) * u.kg
MERCURY_MASS = (3.285 * 10**23) * u.kg
VENUS_MASS = (4.867 * 10**24) * u.kg
SOLAR_SYS_MASS = SUN_MASS + EARTH_MASS + MARS_MASS + JUPITER_MASS + \
    SATURN_MASS + URANAS_MASS + NEPTUNE_MASS + MERCURY_MASS + VENUS_MASS
SOLAR_SYS_MU = SOLAR_SYS_MASS * G

# Intialize bodies
epoch = Time("2026-11-08")
solar_system_ephemeris.set('de432s')
sun = Body(SUN_MASS, epoch, celestial_body='sun', color="yellow")
earth = Body(EARTH_MASS, epoch, celestial_body="earth", color="green")
earth.mu = G.value * earth.mass
moon = Body(MOON_MASS, epoch, celestial_body="moon", color='grey')
moon.mu = G.value * moon.mass
mars = Body(MARS_MASS, epoch, celestial_body="mars", color="red")
mars.mu = G.value * mars.mass
jupiter = Body(JUPITER_MASS, epoch, celestial_body="jupiter", color="orange")
jupiter.mu = G.value * jupiter.mass
saturn = Body(SATURN_MASS, epoch, celestial_body="saturn", color="yellow")
saturn.mu = G.value * saturn.mass
uranus = Body(URANAS_MASS, epoch, celestial_body="uranus", color="cyan")
uranus.mu = G.value * uranus.mass
neptune = Body(NEPTUNE_MASS, epoch, celestial_body="neptune", color="cyan")
neptune.mu = G.value * neptune.mass
mercury = Body(MERCURY_MASS, epoch, celestial_body="mercury", color="cyan")
mercury.mu = G.value * mercury.mass
venus = Body(VENUS_MASS, epoch, celestial_body="venus", color="cyan")
venus.mu = G.value * venus.mass
