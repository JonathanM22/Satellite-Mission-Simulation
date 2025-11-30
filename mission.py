"""
n-body. Trying to do n-body propagation
"""
from orbit import *
from Orbit_util import *
from body import *
import numpy as np
import matplotlib.pyplot as plt
import time

from astropy.time import Time
from astropy.time import TimeDelta
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
from astropy.coordinates import get_body_barycentric

"""
Functions
"""


def RK4_single_step(fun, dt, t0, y0, fun_arg: list):

    k1 = fun(t0, y0, fun_arg)
    k2 = fun((t0 + (dt/2)), (y0 + ((dt.value/2)*k1)), fun_arg)
    k3 = fun((t0 + (dt/2)), (y0 + ((dt.value/2)*k2)), fun_arg)
    k4 = fun((t0 + dt), (y0 + (dt.value*k3)), fun_arg)

    y1 = y0 + (dt.value/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y1


def y_dot_n_ephemeris(t, y, fun_arg: list):
    """
        t: astropy time object
        y: np.array
        fun_arg: premade  

        direction matters!
        r = distance from central body -> sat
        r_c = distance from origin -> central body
        r_k = distance from origin -> kth body
        r_sk = distance from sat -> kth body
        r_s = 

        m_c = central body mass
        m_k = kth body mass
    """
    fun_arg[0] = central_body
    fun_arg[1] = bodies

    r = y[0:3]
    r_mag = np.linalg.norm(r)
    v = y[3:6]

    r_c = get_body_barycentric(central_body.label, t).xyz.to(u.km).value
    m_c = central_body.mass.value

    a = ((G.value*m_c)/(r_mag**3)) * -r

    for body in bodies:
        r_k = get_body_barycentric(body.label, t).xyz.to(u.km).value
        m_k = body.mass.value

        # r_s = r_c + r
        # r_sk = r_k - r_s

        r_ck = r_k - r_c
        r_sk = r_ck - r

        r_sk_mag = np.linalg.norm(r_sk)
        a = a + (((G.value*m_k)/(r_sk_mag**3)) * r_sk)

    y_dot = np.concatenate((v, a))

    return y_dot


"""
Constants and Intialization
"""
program_start_timer = time.perf_counter()

MOON_MASS = (7.34 * 10**22) * u.kg
MARS_MASS = (6.39 * 10**23) * u.kg
SAT_MASS = 100 * u.kg
G = const.G.to(u.km**3 / (u.kg * u.s**2))  # convert to km
SUN_MU = const.GM_sun.to(u.km**3 / u.s**2)
EARTH_MU = const.GM_earth.to(u.km**3 / u.s**2)

# Intialize bodies
epoch = Time("2026-11-08")
solar_system_ephemeris.set('de432s')
sun = Body(const.M_sun, epoch, celestial_body='sun', color="yellow")
earth = Body(const.M_earth, epoch, celestial_body="earth", color="green")
moon = Body(MOON_MASS, epoch, celestial_body="moon", color='grey')
mars = Body(MARS_MASS, epoch, celestial_body="mars", color="red")

# Intialize SAT
SAT_MASS = 100*u.kg
sat = Spacecraft(SAT_MASS, epoch, label="S/C", color="purple")

# Define Parking Orbit
earth_parking = Orbit(mu=EARTH_MU,
                      a=32000*u.km,
                      e=0.80*u.km/u.km,  # unitless
                      f0=(180*u.deg).to(u.rad),
                      inc=(28*u.deg).to(u.rad),
                      raan=(175*u.deg).to(u.rad),
                      aop=(240*u.deg).to(u.rad)
                      )

earth_parking.p = earth_parking.calc_p(earth_parking.a, earth_parking.e)
earth_parking.energy = earth_parking.calc_energy(
    earth_parking.a, earth_parking.mu)

transfer_short = Orbit(mu=const.GM_sun)

"""
Propagating the earth parking orbit
"""
central_body = earth
sat_orbit = earth_parking
bodies = [moon]

# Getting intial position & vel
r = sat_orbit.r_at_true_anomaly(sat_orbit.e, sat_orbit.p, sat_orbit.f0)

# Everything is in km, for numpy to work you need to have float numbers
r_pqw, v_pqw = orb_2_pqw(r.value,
                         sat_orbit.f0.value, sat_orbit.e.value,
                         sat_orbit.p.value, sat_orbit.mu.value)

r_eci, v_eci = perif_2_eci(r_pqw, v_pqw, sat_orbit.inc.value,
                           sat_orbit.raan.value, sat_orbit.aop.value)

sat.r0 = r_eci * u.km
sat.v0 = v_eci * (u.km/u.s)

y0 = np.concatenate((sat.r0.value, sat.v0.value))

# Intiate Solver
propagation_start_timer = time.perf_counter()
t0 = epoch
# tf = t0 + sat_orbit.period(sat_orbit.a, sat_orbit.mu)*3
tf = t0 + TimeDelta(30, format='jd')
dt = TimeDelta(60, format='sec')
ts = np.arange(t0, tf, dt)
n_steps = len(ts)
ys = np.zeros((n_steps, 6))
ys[0] = y0
ts[0] = t0
fun_arg = [central_body, bodies]

step = 1
for i in range(len(ts) - 1):
    ys[step] = RK4_single_step(
        y_dot_n_ephemeris, dt, ts[step-1], ys[step-1], fun_arg=fun_arg)
    step += 1
propagation_time = time.perf_counter() - propagation_start_timer


# Fill out bodies arrays for plotting
central_body.t_ar = ts
for body in bodies:
    body.t_ar = ts

central_body.r_ar = np.zeros((n_steps, 3))
sat.r_ar = np.zeros((n_steps, 3))
for body in bodies:
    body.r_ar = np.zeros((n_steps, 3))

for i, t in enumerate(ts):
    central_body.r_ar[i] = get_body_barycentric(
        central_body.label, t).xyz.to(u.km).value
    for body in bodies:
        body.r_ar[i] = get_body_barycentric(body.label, t).xyz.to(u.km).value

# Generating position of sat in terms of barycenter
for i in range(len(ts)):
    # r_s = r_c + r
    sat.r_ar[i] = central_body.r_ar[i] + ys[i, 0:3]

program_time = time.perf_counter() - program_start_timer
print(f'------------------------------------------------------------------------')
print(f'Propagation took {propagation_time} seconds')
print(f'Python Script took {program_time} seconds')
print(f'------------------------------------------------------------------------')


"""
Plotting
"""
plot = True

if plot:
    """
    Plot trajectory 
    """

    earth_frame = True
    if earth_frame:
        ax = plt.figure().add_subplot(projection='3d')

        # Add Earth
        ax.scatter(0, 0, 0, color=central_body.color, s=5,
                   marker='o', edgecolor='k', label=central_body.label)

        moon_wrt_earth = moon.r_ar - earth.r_ar

        ax.plot(moon_wrt_earth[:, 0],
                moon_wrt_earth[:, 1],
                moon_wrt_earth[:, 2],
                color=moon.color,
                label=moon.label)

        ax.plot(ys[:, 0], ys[:, 1], ys[:, 2],
                color=sat.color,
                label=sat.label)

        ax.set_title(f"Earth Frame", fontsize=14, pad=10)
        ax.set_aspect('equal')
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_zlabel("Z [km]")
        ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
        plt.tight_layout()
        ax.view_init(elev=np.rad2deg(sat_orbit.inc.value),
                     azim=np.rad2deg(sat_orbit.raan.value), roll=0)
        plt.show()

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(central_body.r_ar[:, 0],
            central_body.r_ar[:, 1],
            central_body.r_ar[:, 2],
            color=central_body.color,
            label=central_body.label)

    ax.plot(moon.r_ar[:, 0],
            moon.r_ar[:, 1],
            moon.r_ar[:, 2],
            color=moon.color,
            label=moon.label)

    ax.plot(sat.r_ar[:, 0], sat.r_ar[:, 1], sat.r_ar[:, 2],
            color=sat.color, label=sat.label)

    ax.set_title(f"Barycenter Frame", fontsize=14, pad=10)
    ax.set_aspect('equal')
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    plt.tight_layout()
    plt.show()

    """ Plot orbital stats """

    # Keeping track of all important elements for RK4
    h2 = np.zeros(np.size(ys[:, 0]))
    a2 = np.zeros(np.shape(h2))
    e2 = np.zeros(np.shape(h2))
    inc2 = np.zeros(np.shape(h2))
    raan2 = np.zeros(np.shape(h2))
    aop2 = np.zeros(np.shape(h2))

    for i in range(np.size(h2)-1):
        a, e, e_vec, inc, raan, aop, f = rv_2_orb_elm(
            ys[i, 0:3], ys[i, 0:3], sat_orbit.mu.value)

        h2[i] = np.linalg.norm(np.cross(ys[i], ys[i]))
        a2[i] = a
        e2[i] = e
        inc2[i] = inc
        raan2[i] = raan
        aop2[i] = aop

        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(h1[0:-1], color='orange', label='scipy')
        ax1.plot(h2[0:-1], color='blue', linestyle='--', label='rk4')
        ax1.set_xlabel("time (seconds)")
        ax1.set_ylabel("h")
        ax1.set_title("Angular Momentum Comparison")
        ax1.legend(loc='lower left')
        ax1.grid(True)

        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(a1[0:-1], color='orange', label='scipy')
        ax2.plot(a2[0:-1], color='blue', linestyle='--', label='rk4')
        ax2.set_xlabel("time (seconds)")
        ax2.set_ylabel("a")
        ax2.set_title("SMA Comparison")
        ax2.legend(loc='lower left')
        ax2.grid(True)

        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(e1[0:-1], color='orange', label='scipy')
        ax3.plot(e2[0:-1], color='blue', linestyle='--', label='rk4')
        ax3.set_xlabel("time (seconds)")
        ax3.set_ylabel("e")
        ax3.set_title("ECC Comparison")
        ax3.legend(loc='lower left')
        ax3.grid(True)

        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(inc1[0:-1], color='orange', label='scipy')
        ax4.plot(inc2[0:-1], color='blue', linestyle='--', label='rk4')
        ax4.set_xlabel("time (seconds)")
        ax4.set_ylabel("inc")
        ax4.set_title("INC Comparison")
        ax4.legend(loc='lower left')
        ax4.grid(True)

        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(raan1[0:-1], color='orange', label='scipy')
        ax5.plot(raan2[0:-1], color='blue', linestyle='--', label='rk4')
        ax5.set_xlabel("time (seconds)")
        ax5.set_ylabel("raan")
        ax5.set_title("RAAN Comparison")
        ax5.legend(loc='lower left')
        ax5.grid(True)

        ax6 = plt.subplot(3, 2, 6)
        ax6.plot(aop1[0:-1], color='orange', label='scipy')
        ax6.plot(aop2[0:-1], color='blue', linestyle='--', label='rk4')
        ax6.set_xlabel("time (seconds)")
        ax6.set_ylabel("aop")
        ax6.set_title("AOP Comparison")
        ax6.legend(loc='lower left')
        ax6.grid(True)

        plt.legend()
        plt.tight_layout()
        plt.show()
