"""
n-body. Trying to do n-body propagation
"""

# Custom libs
from orbit import *
from Orbit_util import *
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
print("\n----------------Started Simulation----------------\n")

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
SOLAR_SYS_MASS = SUN_MASS + EARTH_MASS + MARS_MASS + JUPITER_MASS + \
    SATURN_MASS + URANAS_MASS + NEPTUNE_MASS + MERCURY_MASS + VENUS_MASS
SOLAR_SYS_MU = SOLAR_SYS_MASS * G

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

# Bodies to save for entire mission plotter
celestial_bodies = [sun, earth, moon, mars, mercury,
                    jupiter, venus, saturn, uranus, neptune]

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
mars_parking = Orbit(mu=MARS_MU,
                     a=MARS_RAD+2000*u.km,
                     e=0*u.km/u.km,  # unitless
                     f0=(180*u.deg).to(u.rad),
                     inc=(45*u.deg).to(u.rad),
                     raan=(0*u.deg).to(u.rad),
                     aop=(0*u.deg).to(u.rad)
                     )

mars_parking.p = mars_parking.calc_p(mars_parking.a, mars_parking.e)

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
ts = np.arange(t0, tf+dt, dt)
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

# Intiate Solver
propagation_start_timer_2 = time.perf_counter()
dt = TimeDelta(86400, format='sec')
t0 = sat.t_ar[-1]
tf = t0 + tof
ts = np.arange(t0, tf + dt, dt)
n_steps_2 = len(ts)

# Solve Lamberts
r1_lambert = sat.r_ar[-1]
r2_lambert = get_body_barycentric(
    mars.label, arrival_date).xyz.to(u.km).value

mars_miss = r2_lambert

# Uncomment if you have already a solved value
r2_lambert = np.array([-2.55889197e+08, -8.78471051e+06, -4.35998868e+06])

while (np.linalg.norm(mars_miss) > mars_parking.a.value):

    a, p, e, tranfer_v1, tranfer_v2 = universal_lambert(
        r1_lambert, r2_lambert, tof.sec, tranfer_orbit.mu.value, desired_path='long')
    tranfer_orbit.a = a * u.km
    tranfer_orbit.e = e * u.km/u.km
    tranfer_orbit.p = p * u.km

    # Calculting intial pos of sat in terms of central body
    r_s = sat.r_ar[-1]
    r_c = get_body_barycentric(central_body.label, t0).xyz.to(u.km).value
    r0 = r_s - r_c
    dv1 = tranfer_v1 - sat.v_ar[-1]
    v0 = tranfer_v1
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

    # r = r_s - r_c
    mars_miss = sat_pos_bary[-1] - \
        get_body_barycentric(mars.label, arrival_date).xyz.to(u.km).value

    if np.linalg.norm(mars_miss) > 100000:
        r2_lambert = r2_lambert - mars_miss*0.50
    elif np.linalg.norm(mars_miss) > 15000:
        r2_lambert = r2_lambert - mars_miss*0.25
    else:
        r2_lambert = r2_lambert - mars_miss*0.10

    print(f"Sat is {np.linalg.norm(mars_miss)} km away from mars")

print(f"r2 is: {r2_lambert}")
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
dt_steps = np.round(
    (sat_orbit.period(sat_orbit.a, sat_orbit.mu).value*2)/dt.value)
t0 = sat.t_ar[-1]
tf = t0 + dt_steps * dt
# tf = t0 + TimeDelta(86400*1, format='sec')
ts = np.arange(t0, tf + dt, dt)
n_steps_3 = len(ts)

# Calculting intial pos of sat in terms of central body
r_s = sat.r_ar[-1]
v_s = sat.v_ar[-1]
r, v = get_body_barycentric_posvel(central_body.label, t0)
r_c = r.xyz.to(u.km).value
v_c = v.xyz.to(u.km/u.s).value
r_rel_mars = r_s - r_c
v_rel_mars = v_s - v_c

# Re-defining semi-major axis cause techincally our propagation did not get the desired distance.
# Can just set it equal to the position of the sat relative to CB cause in circle orbit
# semi-major axis = radial distance (a=r)
sat_orbit.a = np.linalg.norm(r_rel_mars) * u.km

# Assuming mars parking orbit v = sqrt(mu/a)
v0 = np.sqrt(sat_orbit.mu.value/sat_orbit.a.value)

# Need to make deltaV in same direction as sat vel
# Why? cause I'm assume r_rel_mars cross v_rel_mars defines a orbital plane
# and that plane will be were our parking orbit will be in.
vel_dir = v_rel_mars / np.linalg.norm(v_rel_mars)

r0 = r_rel_mars
v0 = vel_dir * v0
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

for celestial_body in celestial_bodies:
    celestial_body.r_ar = np.zeros((len(sat.t_ar), 3))
    celestial_body.v_ar = np.zeros((len(sat.t_ar), 3))
    for i, t in enumerate(sat.t_ar):
        r, v = get_body_barycentric_posvel(celestial_body.label, t)

        celestial_body.r_ar[i] = r.xyz.to(u.km).value
        celestial_body.v_ar[i] = v.xyz.to(u.km/u.s).value

np.savez("mission_data", np.array(sat), celestial_bodies)
print(f'------------------------------------------------------------------------')
print(f'Leg 1 Propagation took {propagation_time_1} seconds')
print(f'Leg 2 Propagation took {propagation_time_2} seconds')
print(f'Leg 3 Propagation took {propagation_time_3} seconds')
print(
    f'Python Script took {time.perf_counter() - program_start_timer} seconds')
print(f'------------------------------------------------------------------------')
