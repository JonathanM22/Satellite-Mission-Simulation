"""
n-body. Trying to do n-body propagation
"""
from orbit import *
from Orbit_util import *
from body import *
import numpy as np
from scipy.integrate import ode
from scipy import optimize
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.time import TimeDelta
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel

"""
Constants and Intialization
"""
MOON_MASS = (7.34 * 10**22) * u.kg
MARS_MASS = (6.39 * 10**23) * u.kg
SAT_MASS = 100 * u.kg
G = const.G.to(u.km**3 / (u.kg * u.s**2))  # convert to km
SUN_MU = const.GM_sun.to(u.km**3 / u.s**2)
EARTH_MU = const.GM_earth.to(u.km**3 / u.s**2)

# Intialize epoch, tof & arrivial date
epoch = Time("2026-11-08")
tof = TimeDelta(180, format="jd")
arrival_date = epoch + tof

# Intialize bodies
sun = Body(const.M_sun, epoch, celestial_body='sun', color="yellow")
earth = Body(const.M_earth, epoch, celestial_body="earth", color="green")
moon = Body(MOON_MASS, epoch, celestial_body="moon", color='grey')
mars = Body(MARS_MASS, epoch, celestial_body="mars", color="red")

SAT_MASS = 100*u.kg
sat = Spacecraft(SAT_MASS, epoch, label="S/C", color="purple")

earth_parking = Orbit(mu=EARTH_MU,
                      a=32000*u.km,
                      e=0.95*u.km/u.km,
                      f0=(180*u.deg).to(u.rad),
                      inc=(28*u.deg).to(u.rad),
                      raan=(174*u.deg).to(u.rad),
                      aop=(240*u.deg).to(u.rad)
                      )

earth_parking.p = earth_parking.calc_p()
earth_parking.energy = earth_parking.calc_energy()

transfer_short = Orbit(mu=const.GM_sun)

"""
Propagating the earth parking orbit
"""
central_body = earth
sat_orbit = earth_parking
bodies = [moon]

# Getting intial position & vel
r = sat_orbit.r_at_true_anomaly(f=sat_orbit.f0)

# Everything is in km, for numpy to work you need to have float numbers
r_pqw, v_pqw = orb_2_pqw(r.value,
                         sat_orbit.f0.value, sat_orbit.e.value,
                         sat_orbit.p.value, sat_orbit.mu.value)

r_eci, v_eci = perif_2_eci(r_pqw, v_pqw, sat_orbit.inc.value,
                           sat_orbit.raan.value, sat_orbit.aop.value)

sat.r0 = central_body.r0 + r_eci * u.km
sat.v0 = central_body.v0 + v_eci * (u.km/u.s)

y0 = np.concatenate((sat.r0.value, sat.v0.value))

# Intiate Solver
dt = 3600/2
tspan = sat_orbit.period() * 1.1
n_steps = int(np.ceil(tspan/dt))  # type:ignore
ys = np.zeros((n_steps, 6))
ts = np.zeros((n_steps, 1))

solver = ode(y_dot_n_ephemeris)
solver.set_integrator('lsoda')
solver.set_initial_value(y0, 0)
solver.set_f_params(central_body)
step = 1

while solver.successful() and step < n_steps:

    solver.integrate(solver.t+dt)
    ts[step, :] = solver.t
    ys[step, :] = solver.y
    step += 1


def y_dot_n_ephemeris(t, y, central_body: Body, bodies: list[Body], epoch: Time):
    """
        Direction matters!
        r = distance from central body -> sat
        r_c = distance from origin -> central body
        r_k = distance from origin -> kth body
        r_sk = distance from sat -> kth body

        m_c = central body mass
        m_k = kth body mass
    """

    y_dot = np.zeros(n_bodies*6)
    current_time = epoch + TimeDelta(t, format="sec")

    # Central Body posvel
    m_c = central_body.mass.value
    r_c, v_c = get_body_barycentric_posvel(
        central_body.celestial_body, current_time)
    r_c = r_c.xyz.to(u.km).value
    v_c = v_c.xyz.to(u.km/u.s).value

    # Sat posvel
    rs = y_nbody[0:3]
    vs = y_nbody[3:6]

    r = rs - r_c
    r_mag = np.linalg.norm(r)

    a = ((G*m_c)/(r_mag**3)) * -r

    for body in bodies:

        # Kth Body posvel
        m_k = central_body.mass.value
        r_k, v_k = get_body_barycentric_posvel(
            body.celestial_body, current_time)
        r_k = r_k.xyz.to(u.km).value
        v_k = v_k.xyz.to(u.km/u.s).value

        r_sk = r_k - rs
        r_sk_mag = np.linalg.norm(r_sk)
        m_k = body.mass.value

        a += ((G*m_k)/(r_sk_mag**3)) * r_sk

    y_dot = np.concatenate(vs, a)

    return y_dot


"""
Plotting
"""

plot = True

if plot:
    """
    Plot trajectory 
    """
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

    """
    Plot Distance between S/C and Mars
    """
    dist_vec = sat.r_ar - mars.r_ar  # type:ignore
    dist_mag = np.linalg.norm(dist_vec, axis=1)

    # Plot distance vs time
    plt.figure()
    plt.plot(ts[:, 0] / 86400, dist_mag / 1e6)
    plt.xlabel("Time [days]")
    plt.ylabel("Distance between S/C and Mars [million km]")
    plt.title("Separation between Satellite and Mars over Time")
    plt.grid(True)
    plt.show()
