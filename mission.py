"""
n-body. Trying to do n-body propagation
"""

# Custom libs
from orbit import *
from orbit_util import *
from body import *
from Universal_Variable import *

# Standard libs
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

# #solar_system_ephemeris.bodies
# ('earth', 'sun', 'moon', 'mercury', 'venus', 'earth-moon-barycenter', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')


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

    # print(f'Accel from CB: {a}')

    for body in bodies:
        r_k = get_body_barycentric(body.label, t).xyz.to(u.km).value
        m_k = body.mass.value

        # r_s = r_c + r
        # r_sk = r_k - r_s

        r_ck = r_k - r_c
        r_sk = r_ck - r

        r_sk_mag = np.linalg.norm(r_sk)

        a_k = (((G.value*m_k)/(r_sk_mag**3)) * r_sk)
        # print(f'Accel from {body.label}: {a_k}')

        a = a + a_k

    y_dot = np.concatenate((v, a))

    return y_dot


"""
Constants and Intialization
"""
program_start_timer = time.perf_counter()

MOON_MASS = (7.34 * 10**22) * u.kg
SAT_MASS = 100 * u.kg
G = const.G.to(u.km**3 / (u.kg * u.s**2))  # convert to km
SUN_MASS = const.M_sun
SUN_MU = const.GM_sun.to(u.km**3 / u.s**2)
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

# Intialize bodies
epoch = Time("2026-11-08")
solar_system_ephemeris.set('de432s')
sun = Body(SUN_MASS, epoch, celestial_body='sun', color="yellow")
earth = Body(EARTH_MASS, epoch, celestial_body="earth", color="green")
moon = Body(MOON_MASS, epoch, celestial_body="moon", color='grey')
mars = Body(MARS_MASS, epoch, celestial_body="mars", color="red")
jupiter = Body(JUPITER_MASS, epoch, celestial_body="jupiter", color="orange")
saturn = Body(SATURN_MASS, epoch, celestial_body="saturn", color="yellow")
uranus = Body(URANAS_MASS, epoch, celestial_body="uranus", color="cyan")
neptune = Body(NEPTUNE_MASS, epoch, celestial_body="neptune", color="cyan")
mercury = Body(MERCURY_MASS, epoch, celestial_body="mercury", color="cyan")
venus = Body(VENUS_MASS, epoch, celestial_body="venus", color="cyan")


celestial_bodies = [sun, earth, moon, mars]

# Intialize SAT
SAT_MASS = 100*u.kg
sat = Spacecraft(SAT_MASS, epoch, label="S/C", color="purple")

# Define Earth Parking Orbit
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

# Defne Mars Parking Orbit
mars_parking = Orbit(mu=MARS_MU)

"""
############ LEG-1: Propagating the earth parking orbit ############
"""
print("\nLEG-1: Propagating the earth parking orbit\n")
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
propagation_start_timer_1 = time.perf_counter()
dt = TimeDelta(60, format='sec')
dt_steps = np.round(
    (sat_orbit.period(sat_orbit.a, sat_orbit.mu).value*2)/dt.value)
t0 = epoch - dt*dt_steps + dt
tf = epoch
ts = np.arange(t0, tf, dt)
n_steps_1 = len(ts)
ys = np.zeros((n_steps_1, 6))
ys[0] = y0
ts[0] = t0
fun_arg = [central_body, bodies]

step = 1
for i in range(len(ts) - 1):
    ys[step] = RK4_single_step(
        y_dot_n_ephemeris, dt, ts[step-1], ys[step-1], fun_arg=fun_arg)
    step += 1
propagation_time_1 = time.perf_counter() - propagation_start_timer_1

# Generating position of SAT in terms of barycenter
central_body.r_ar = np.zeros((n_steps_1, 3))
central_body.v_ar = np.zeros((n_steps_1, 3))
sat_pos_bary = np.zeros((n_steps_1, 3))
sat_vel_bary = np.zeros((n_steps_1, 3))
for i, t in enumerate(ts):
    r, v = get_body_barycentric_posvel(
        central_body.label, t)

    central_body.r_ar[i] = r.xyz.to(u.km)
    central_body.v_ar[i] = v.xyz.to(u.km/u.s)

for i in range(len(ts)):
    sat_pos_bary[i] = central_body.r_ar[i] + ys[i, 0:3]   # r_s = r_c + r
    sat_vel_bary[i] = central_body.v_ar[i] + ys[i, 3:6]

sat.r_ar = sat_pos_bary
sat.v_ar = sat_vel_bary
sat.t_ar = ts

# Saving all the data for LEG-1
data_dict = {
    "central_body": central_body,
    "sat_orbit": sat_orbit,
    "bodies": bodies,
    "dt": dt,
    "n_steps": n_steps_1,
    "t0": t0,
    "tf": tf,
    "y0": y0,
    "ts": ts,
    "ys": ys
}

leg_1_data = np.save("leg_1_data", np.array(data_dict))

"""
############ LEG-2: Propagating the earth mars Transfer ############
"""
print("\nLEG-2: Propagating the earth mars Transfer\n")
# Defining Transfer Orbit
tof = TimeDelta(180, format='jd')
arrival_date = epoch + tof
tranfer_orbit = Orbit(mu=SUN_MU)

central_body = sun
sat_orbit = tranfer_orbit
target_body = mars
bodies = [earth, mars, mercury, jupiter, venus, saturn, uranus, neptune]

# Earth @ depature and Mars @ arrival helio centric
# Helito centric cords is ONLY FOR THE LAMBERT PROBLEM!!
r1_sun_lambert = get_body_barycentric(sun.label, epoch).xyz.to(u.km).value
r2_sun_lambert = get_body_barycentric(
    sun.label, arrival_date).xyz.to(u.km).value
r1_lambert = get_body_barycentric(
    earth.label, epoch).xyz.to(u.km).value - r1_sun_lambert
r2_lambert = get_body_barycentric(
    mars.label, arrival_date).xyz.to(u.km).value - r2_sun_lambert

a, p, e, tranfer_v1, tranfer_v2 = universal_lambert(
    r1_lambert, r2_lambert, tof.sec, tranfer_orbit.mu.value, desired_path='long')

tranfer_orbit.a = a * u.km
tranfer_orbit.e = e * u.km/u.km
tranfer_orbit.p = p * u.km

# Intiate Solver
propagation_start_timer_2 = time.perf_counter()
dt = TimeDelta(86400, format='sec')
t0 = sat.t_ar[-1]
tf = t0 + tof
ts = np.arange(t0, tf, dt)
n_steps_2 = len(ts)

# Calculting intial pos of sat in terms of central body
r_s = sat.r_ar[-1]
r_c = get_body_barycentric(central_body.label, t0).xyz.to(u.km).value
r0 = r_s - r_c
v0 = tranfer_v1*2
y0 = np.concatenate((r0, v0))

ys = np.zeros((n_steps_2, 6))
ys[0] = y0
ts[0] = t0
fun_arg = [central_body, bodies]

step = 1
for i in range(len(ts) - 1):
    ys[step] = RK4_single_step(
        y_dot_n_ephemeris, dt, ts[step-1], ys[step-1], fun_arg=fun_arg)
    step += 1
propagation_time_2 = time.perf_counter() - propagation_start_timer_2


# Generating position of sat in terms of barycenter
central_body.r_ar = np.zeros((n_steps_2, 3))
central_body.v_ar = np.zeros((n_steps_2, 3))
sat_pos_bary = np.zeros((n_steps_2, 3))
sat_vel_bary = np.zeros((n_steps_2, 3))
for i, t in enumerate(ts):
    r, v = get_body_barycentric_posvel(
        central_body.label, t)

    central_body.r_ar[i] = r.xyz.to(u.km)
    central_body.v_ar[i] = v.xyz.to(u.km/u.s)

for i in range(len(ts)):
    sat_pos_bary[i] = central_body.r_ar[i] + ys[i, 0:3]   # r_s = r_c + r
    sat_vel_bary[i] = central_body.v_ar[i] + ys[i, 3:6]

sat.r_ar = np.concatenate((sat.r_ar, sat_pos_bary))
sat.v_ar = np.concatenate((sat.v_ar, sat_vel_bary))
sat.t_ar = np.concatenate((sat.t_ar, ts))

# Saving all the data for LEG-2
data_dict = {
    "central_body": central_body,
    "sat_orbit": sat_orbit,
    "bodies": bodies,
    "dt": dt,
    "n_steps": n_steps_2,
    "t0": t0,
    "tf": tf,
    "y0": y0,
    "ts": ts,
    "ys": ys
}

leg_2_data = np.save("leg_2_data", np.array(data_dict))


"""
############ LEG-3: Propagating Mars insertion ############
"""
print("\nLEG-3: Propagating Mars insertion\n")

central_body = mars
sat_orbit = mars_parking
bodies = []

# Intiate Solver
propagation_start_timer_3 = time.perf_counter()
dt = TimeDelta(180, format='sec')
t0 = sat.t_ar[-1]
tf = t0 + TimeDelta(86400*3, format='sec')
ts = np.arange(t0, tf, dt)
n_steps_3 = len(ts)

# Calculting intial pos of sat in terms of central body
r_s = sat.r_ar[-1]
r, v = get_body_barycentric_posvel(central_body.label, t0)
r_c = r.xyz.to(u.km).value
v_c = v.xyz.to(u.km/u.s).value
r0 = r_s - r_c
# UNSURE ABOUT THIS!!!!!
v0 = (sat.v_ar[-1] - v_c) - v_c  # UNSURE ABOUT THIS!!!!!
y0 = np.concatenate((r0, v0))

ys = np.zeros((n_steps_3, 6))
ys[0] = y0
ts[0] = t0
fun_arg = [central_body, bodies]

step = 1
for i in range(len(ts) - 1):
    ys[step] = RK4_single_step(
        y_dot_n_ephemeris, dt, ts[step-1], ys[step-1], fun_arg=fun_arg)
    step += 1
propagation_time_3 = time.perf_counter() - propagation_start_timer_3

# Generating position of sat in terms of barycenter
central_body.r_ar = np.zeros((n_steps_3, 3))
central_body.v_ar = np.zeros((n_steps_3, 3))
sat_pos_bary = np.zeros((n_steps_3, 3))
sat_vel_bary = np.zeros((n_steps_3, 3))
for i, t in enumerate(ts):
    r, v = get_body_barycentric_posvel(
        central_body.label, t)

    central_body.r_ar[i] = r.xyz.to(u.km)
    central_body.v_ar[i] = v.xyz.to(u.km/u.s)

for i in range(len(ts)):
    sat_pos_bary[i] = central_body.r_ar[i] + ys[i, 0:3]   # r_s = r_c + r
    sat_vel_bary[i] = central_body.v_ar[i] + ys[i, 3:6]

sat.r_ar = np.concatenate((sat.r_ar, sat_pos_bary))
sat.v_ar = np.concatenate((sat.v_ar, sat_vel_bary))
sat.t_ar = np.concatenate((sat.t_ar, ts))


# Saving all the data for LEG-2
data_dict = {
    "central_body": central_body,
    "sat_orbit": sat_orbit,
    "bodies": bodies,
    "dt": dt,
    "n_steps": n_steps_3,
    "t0": t0,
    "tf": tf,
    "y0": y0,
    "ts": ts,
    "ys": ys
}

leg_3_data = np.save("leg_3_data", np.array(data_dict))


np.savez("mission_data", np.array(sat), celestial_bodies)
print(f'------------------------------------------------------------------------')
print(f'Leg 1 Propagation took {propagation_time_1} seconds')
print(f'Leg 2 Propagation took {propagation_time_2} seconds')
print(f'Leg 3 Propagation took {propagation_time_3} seconds')
print(
    f'Python Script took {time.perf_counter() - program_start_timer} seconds')
print(f'------------------------------------------------------------------------')


# for body in bodies:
#     body.t_ar = ts

# central_body.r_ar = np.zeros((n_steps, 3))
# sat.r_ar = np.zeros((n_steps, 3))
# for body in bodies:
#     body.r_ar = np.zeros((n_steps, 3))

# for i, t in enumerate(ts):
#     central_body.r_ar[i] = get_body_barycentric(
#         central_body.label, t).xyz.to(u.km).value
#     for body in bodies:
#         body.r_ar[i] = get_body_barycentric(body.label, t).xyz.to(u.km).value

# """
# Plotting
# """
# plot = True

# if plot:

#     """Earth Frame"""
#     earth_frame = True
#     if earth_frame:
#         ax = plt.figure().add_subplot(projection='3d')

#         # Add Earth
#         ax.scatter(0, 0, 0, color=central_body.color, s=5,
#                    marker='o', edgecolor='k', label=central_body.label)

#         moon_wrt_earth = moon.r_ar - earth.r_ar

#         ax.plot(moon_wrt_earth[:, 0],
#                 moon_wrt_earth[:, 1],
#                 moon_wrt_earth[:, 2],
#                 color=moon.color,
#                 label=moon.label)

#         ax.plot(ys[:, 0], ys[:, 1], ys[:, 2],
#                 color=sat.color,
#                 label=sat.label)

#         ax.set_title(f"Earth Frame", fontsize=14, pad=10)
#         ax.set_aspect('equal')
#         ax.set_xlabel("X [km]")
#         ax.set_ylabel("Y [km]")
#         ax.set_zlabel("Z [km]")
#         ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
#         plt.tight_layout()
#         ax.set_xlim([-50000, 50000])
#         ax.set_ylim([-50000, 50000])
#         ax.set_zlim([-50000, 50000])
#         ax.view_init(elev=np.rad2deg(sat_orbit.inc.value),
#                      azim=90, roll=0)
#         plt.show()

#     """ Bary Centric Frame """
#     ax = plt.figure().add_subplot(projection='3d')
#     ax.plot(central_body.r_ar[:, 0],
#             central_body.r_ar[:, 1],
#             central_body.r_ar[:, 2],
#             color=central_body.color,
#             label=central_body.label)

#     ax.plot(moon.r_ar[:, 0],
#             moon.r_ar[:, 1],
#             moon.r_ar[:, 2],
#             color=moon.color,
#             label=moon.label)

#     ax.plot(sat.r_ar[:, 0], sat.r_ar[:, 1], sat.r_ar[:, 2],
#             color=sat.color, label=sat.label)

#     ax.set_title(f"Barycenter Frame", fontsize=14, pad=10)
#     ax.set_aspect('equal')
#     ax.set_xlabel("X [km]")
#     ax.set_ylabel("Y [km]")
#     ax.set_zlabel("Z [km]")
#     ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
#     plt.tight_layout()
#     plt.show()
