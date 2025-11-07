"""
n-body. Trying to do n-body propagation
"""
from orbit import *
from orbit_util import *
from body import *
import numpy as np
from scipy.integrate import ode
from scipy import optimize
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.time import TimeDelta
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
from astropy import constants as const

"""
Constants and Intialization  
"""

# Constants
# ALL constants are in SI UNITS! (meters, seconds, etc.)
# Also for formattng constants are ALL_CAPS
G = 6.6743 * 10**-11
AU = 1.496 * 10**11
SUN_MASS = 1.988 * 10**30  # kg
EARTH_MASS = 1.3166 * 10**24  # kg
MARS_MASS = 6.39 * 10**23  # kg
SAT_MASS = 100  # kg

# Intialize epoch and tof
epoch = Time("2026-11-08")
tof = TimeDelta(365, format='jd')
arrival_date = epoch + tof

# Intialize bodies
sun = Body(SUN_MASS, epoch, celestial_body='sun')
earth = Body(EARTH_MASS, epoch, celestial_body='earth', color='green')
mars = Body(MARS_MASS, epoch, celestial_body='mars', color='red')
sat = Body(SAT_MASS, epoch, r0=earth.r0+1000, v0=earth.v0 +
           1000, label='S/C', color='orange')

central_body = sun
bodies = [earth, mars]
n_bodies = len(bodies)

# Calculate r0 and v0 for sattilite


"""
Propagation 
"""

# Time step info
dt = 86400
tspan = tof.sec

# Generate intial state
y0 = np.zeros(n_bodies*6)
i = 0
for body in bodies:
    y0[i:i+6] = np.concatenate((body.r0, body.v0))
    i += 6

# Intiate Solver
n_steps = int(np.ceil(tspan/dt))  # type:ignore
ys = np.zeros((n_steps, n_bodies*6))
ts = np.zeros((n_steps, 1))

solver = ode(y_dot_n_body)
solver.set_integrator('lsoda')
solver.set_initial_value(y0, 0)
solver.set_f_params(central_body, n_bodies, bodies)
step = 1

while solver.successful() and step < n_steps:
    solver.integrate(solver.t+dt)
    ts[step, :] = solver.t
    ys[step, :] = solver.y
    step += 1

print(step)

i = 0
for body in bodies:
    ys_body = ys[:, i:i+6]
    body.r_ar = ys_body[:, 0:3]  # type:ignore
    body.r_ar[0, :] = body.r0
    body.v_ar = ys_body[:, 3:6]
    body.v_ar[0, :] = body.v0  # type:ignore
    i += 6

"""
Plotting
"""

plot = True

if plot:
    ax = plt.figure().add_subplot(projection='3d')

    # Add Sun
    ax.scatter(0, 0, 0,
               color='yellow', s=15, marker='o', edgecolor='k', label="SUN")

    for body in bodies:

        """
        # Depature Point
        ax.scatter(body.r_ar[0, 0], body.r_ar[0, 1], body.r_ar[0, 2],  # type:ignore
                   color=body.color, s=15, marker='o', edgecolor='k', label=body.label)"""

        ax.plot(body.r_ar[:, 0], body.r_ar[:, 1], body.r_ar[:, 2],  # type:ignore
                color=body.color, label=body.label)

    # formatting
    ax.set_title(
        f"Earth–Mars Transfer Orbits {epoch.strftime('%Y-%m-%d')} - {arrival_date.strftime('%Y-%m-%d')}", fontsize=14, pad=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc='right')
    ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))

    plt.show()
