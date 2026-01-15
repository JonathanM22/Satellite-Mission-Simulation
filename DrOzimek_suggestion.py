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
from poliastro.bodies import Sun
from poliastro.iod import vallado
k = Sun.k
"""
Functions
"""

# Sets up single Runge Kutta 4 Step
def RK4_single_step(fun, dt, t0, y0, fun_arg: list):

    # evaluates inputted function, fun, at t0, y0, and inputted args to create 4 constants to solve 1 rk4 step
    # inputted function name --> y_dot_n_ephemeris
    k1 = fun(t0, y0, fun_arg)
    k2 = fun((t0 + (dt/2)), (y0 + ((dt.value/2)*k1)), fun_arg)
    k3 = fun((t0 + (dt/2)), (y0 + ((dt.value/2)*k2)), fun_arg)
    k4 = fun((t0 + dt), (y0 + (dt.value*k3)), fun_arg)
    y1 = y0 + (dt.value/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y1

def propagate_rk4(r0, v0, t0, tf, dt, fun_arg: list):
    # time array equally by dt
    ts = np.arange(t0, tf, dt)
    n_steps = len(ts)
    ys = np.zeros((n_steps, 6))
    y0 = np.concatenate((r0, v0))
    ys[0] = y0
    step = 1
    for i in range(n_steps - 1):
        ys[i+1] = RK4_single_step(y_dot_n_ephemeris,
                                  dt, ts[i], ys[i], fun_arg=fun_arg)
        step += 1
    r = ys[:, :3]
    v = ys[:, 3:6]

    return r, v, ys

# Whole damn function written out by VSCODE

# def newton_raphson_ephem(mu, r0, v0, dt, t0, tol=1e-8, max_iter=1000, fun_arg: list=[]):
#     """
#     Universal Variable Formulation with Ephemeris Data
#     """

#     # Initial guess for chi
#     r0_mag = np.linalg.norm(r0)
#     v0_mag = np.linalg.norm(v0)
#     alpha = 2/r0_mag - (v0_mag**2)/mu
#     chi = np.sqrt(mu) * abs(alpha) * dt

#     # Newton-Raphson Iteration
#     for i in range(max_iter):
#         z = alpha * chi**2
#         S, C = stumpff_functions(z)

#         r = (chi**2 * C) + (np.dot(r0, v0)/np.sqrt(mu)) * chi * (1 - z * S) + r0_mag * (1 - z * C)
#         f = (r - np.sqrt(mu) * dt)

#         if abs(f) < tol:
#             break

#         df_dchi = (chi * (1 - z * S)) + (np.dot(r0, v0)/np.sqrt(mu)) * (1 - z * C)

#         chi = chi - f / df_dchi

#     # Compute final position and velocity vectors
#     f = 1 - (chi**2 / r0_mag) * C
#     g = dt - (1/np.sqrt(mu)) * chi**3 * S

#     r_vec = f * r0 + g * v0

#     fdot = (np.sqrt(mu) / (r0_mag * np.linalg.norm(r_vec))) * chi * (z * S - 1)
#     gdot = 1 - (chi**2 / np.linalg.norm(r_vec)) * C

#     v_vec = fdot * r0 + gdot * v0

#     return r_vec, v_vec

# Two body motion ODE: creating the y_dot function for n-body with ephemeris data
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
        r_s = distance from barycenter -> sat
        m_c = central body mass
        m_k = kth body mass
        r_ck = distance from central body -> kth body
    """
    central_body = fun_arg[0]
    bodies = fun_arg[1]

    # measuring all distances and velocties with respect to the central body
    # r = distance from central body -> sat
    r = y[0:3]
    r_mag = np.linalg.norm(r)
    v = y[3:6]

    # r_c = distance from origin -> central body
    r_c = get_body_barycentric(central_body.label, t).xyz.to(u.km).value
    m_c = central_body.mass.value

    central_body.mu = G.value * m_c
    # acceleration on satellite due to central body
    a = ((central_body.mu)/(r_mag**3)) * -r
    # print(f'Accel from CB: {a}')

    for body in bodies:

        #  # r_s = r_c + r
        # r_sk = r_k - r_s
        # r_k = distance from origin -> kth body

        r_k = get_body_barycentric(body.label, t).xyz.to(u.km).value
        m_k = body.mass.value

        # r_s = distance from barycenter -> sat
        # r_sk = distance from sat -> kth body
        # r_ck = distance from central body -> kth body

        r_ck = r_k - r_c
        r_sk = r_ck - r
        r_sk_mag = np.linalg.norm(r_sk)

        body.mu = G.value * m_k
        # acceleration on satellite due to kth bodies
        a_k = ((body.mu)/(r_sk_mag**3)) * r_sk
        # print(f'Accel from {body.label}: {a_k}')

        # acceleration on CB due to kth body 
        # a_cb_k = ((body.mu)/(np.linalg.norm(r_ck)**3)) * r_ck

        # if r and v were wrt to barycenter, we would just have a = a + a_k ( a_cb_k term only arises since we are wrt central body, which is also being accelerated by other bodies)
        # total acceleration on satellite due to all bodies
        # a = a + a_k - a_cb_k
        a = a + a_k 

        y_dot = np.concatenate((v, a))

    return y_dot

"""
Constants and Intialization
"""
program_start_timer = time.perf_counter()
print("\n----------------Started Simulation----------------\n")

# Bodies to save for entire mission plotter
celestial_bodies = [sun, earth, moon, mars, mercury, jupiter, venus, saturn, uranus, neptune]

# Intialize SAT
SAT_MASS = 100*u.kg
departure_date = Time("2026-11-08")
sat = Spacecraft(SAT_MASS, departure_date, label="sat", color="purple")

'''
STEP 1

Start with Lambert's problem with R1/R2/TOF.  Pull R1 and R2 from and ephemeris file for Earth and Mars.
Iterate on Lambert's problem until you have chosen a solution that you are happy with (e.g. minimum C3 and arrival Vinfinity at Mars).

'''

transfer_short = Orbit(mu=SUN_MU)
transfer_long = Orbit(mu=SUN_MU)

"""
Using JPL data to get postion and velocity of earth (satellite) at departure and mars (target) at arrival)
"""
solar_system_ephemeris.set('de432s')

tof_range = list(range(100, 400, 10))

results = []

for tof_days in tof_range:
    tof = TimeDelta(tof_days, format='jd')
    arrival_date = departure_date + tof
    print(f'{arrival_date}\n')

    # position vector of earth and mars (initial and final satellite positions) wrt to soloar system barycenter
    r1_earth_eci, v1_earth_eci = get_body_barycentric_posvel( 'earth', departure_date)
    r2_mars_eci, v2_mars_eci = get_body_barycentric_posvel('mars', departure_date+tof)

    """
    Need to get sun position and velocity to transform
    earth and mars baycentric cords to helio-centric: sun centered inertial frame
    """

    # Position of Sun at departure and arrival
    r_sun1, v_sun1 = get_body_barycentric_posvel('sun', departure_date)
    r_sun2, v_sun2 = get_body_barycentric_posvel('sun', departure_date+tof)

    """
    heliocentric position and velocity vectors 
    """

    # Position & Velocity of earth (satellite) wrt respect to sun @ Depature
    r1_earth = (r1_earth_eci.xyz - r_sun1.xyz).to(u.km).value  # type:ignore
    v1_earth = (v1_earth_eci.xyz - v_sun1.xyz).to(u.km/u.s).value  # type:ignore

    # Position & Velocity of mars (satellite) wrtrespect to sun @ Depature
    r2_mars = (r2_mars_eci.xyz - r_sun2.xyz).to(u.km).value  # type:ignore
    v2_mars = (v2_mars_eci.xyz - v_sun2.xyz).to(u.km/u.s).value

    # cant get much frmo this since its restricted to 0 - pi
    transfer_angle = (np.acos(np.dot(r1_earth, r2_mars) / (np.linalg.norm(r1_earth) * np.linalg.norm(r2_mars))))
    h = np.cross(r1_earth, r2_mars)
    if h[2] < 0:
        transfer_angle = 2 * np.pi - transfer_angle
        
    # print(f'Transfer angle: {np.degrees(transfer_angle):.1f}°')

    # goes back to what poliastro was talking about: prograde and retrograde

    print(f'-----------------------------------------------------------------------For TOF of {tof_days} days:-----------------------------------------------------------------------n')
    print(f'Earth Position at Depature: {r1_earth} km')
    print(f'Mars Position at Arrival: {r2_mars} km\n')

    bodies =  [earth, venus, mercury, mars, jupiter, saturn, uranus, neptune]  
    central_body = sun
    fun_arg = [central_body, bodies]

    # print("VALLADO FUNCTION")
    # (v1_short, v2_short), = vallado.lambert(k, r1_earth*u.km, r2_mars*u.km, (tof.sec*u.s), short=True)
    # print(f"Departure velocity: {v1_short} | Arrival velocity: {v2_short}\n")

    # print("VRAJ FUNCTION") -> Universal Lambert Formulation
    transfer_short.a, transfer_short.p, transfer_short.e, transfer_v1_short, transfer_v2_short = universal_lambert( r1_earth, r2_mars, (tof.sec), transfer_short.mu, desired_path= 'short')
    # print(f'Short Transfer semi major axis is {transfer_short.a} km -->  {(transfer_short.a/149597870.7)} AU | Eccentricity = {transfer_short.e} | Departure velocity: {transfer_v1} km/s | Arrival velocity: {transfer_v2} km/s\n')
    C3_short = np.linalg.norm(transfer_v1_short - v1_earth)**2  
    transfer_long.a, transfer_long.p, transfer_long.e, transfer_v1_long, transfer_v2_long = universal_lambert( r1_earth, r2_mars, (tof.sec), transfer_long.mu, desired_path= 'long')
    # print(f'Long Transfer semi major axis is {transfer_long.a} km -->  {(transfer_long.a/149597870.7)} AU | Eccentricity = {transfer_long.e} | Departure velocity: {transfer_v1} km/s | Arrival velocity: {transfer_v2} km/s\n')
    C3_long = np.linalg.norm(transfer_v1_long - v1_earth)**2

    if C3_short < C3_long:
        C3 = C3_short   
        transfer_v1 = transfer_v1_short
        transfer_v2 = transfer_v2_short
    else:
        C3 = C3_long
        transfer_v1 = transfer_v1_long
        transfer_v2 = transfer_v2_long

    Vinf_arrival = np.linalg.norm(transfer_v2 - v2_mars)

    results.append({
            'tof_days': tof_days,
            'C3': C3,
            'V1': transfer_v1,
            'V2': transfer_v2,
            'V_earth': v1_earth,
            'V_mars': v2_mars,
            'V_inf_dep': np.sqrt(C3),
            'Vinf_arrival': Vinf_arrival,
            'r1': r1_earth,
            'r2': r2_mars,
            'arrival_date': arrival_date,
            'transfer_angle': np.degrees(transfer_angle)
        })


"""
# Display results in a table
"""
# Syntax was taken from Claude im not gonna lie LOL

# Spike in the p

print("\n" + "="*140)
print(f"{'TOF (days)':<12} {'C3 (km²/s²)':<15} {'V∞ Dep (km/s)':<16} {'V∞ Arr (km/s)':<16} {'Transfer Angle (°)':<20} {'Arrival Date':<20}")
print("="*140)

for res in results:
    print(f"{res['tof_days']:<12} {res['C3']:<15.2f} {res['V_inf_dep']:<16.3f} "
          f"{res['Vinf_arrival']:<16.3f} {res['transfer_angle']:<20.5f} {res['arrival_date'].iso[:10]:<20}")

print("="*140 + "\n")


# Find lowest C3 
min_C3_idx = np.argmin([r['C3'] for r in results])
print(f"Minimum C3 solution: TOF = {results[min_C3_idx]['tof_days']} days, "  f"C3 = {results[min_C3_idx]['C3']:.2f} km²/s²\n")