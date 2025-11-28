"""
main2.py Uses JPL ephemeris data as r1 and r2 to calculate the the lambert problem
"""
import time
from orbit import *
from Orbit_util import *
import numpy as np
from scipy.integrate import ode
from scipy import optimize
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.time import TimeDelta
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
from Universal_Variable import *
from poliastro.iod import izzo
from poliastro.bodies import Sun    
from poliastro.iod import vallado

# Constants
# ALL constants are in SI UNITS! (meters, seconds, etc.)
# Also for formattng constants are ALL_CAPS
EARTH_RAD = 6.371 * 10**6
AU = 1.496 * 10**11
SUN_MU = 1.327 * 10**20
k = Sun.k


# Use Table 1 https://ssd.jpl.nasa.gov/planets/approx_pos.html
earth = Orbit(a=1.00000261*AU,
              e=0.01671123,
              inc=-0.00001531,
              raan=0,
              aop=102.93768193,
              mu=SUN_MU)

mars = Orbit(a=1.52371034*AU,
             e=0.09339410,
             inc=1.84969142,
             raan=49.55953891,
             aop=-23.94362959,
             mu=SUN_MU)


venus = Orbit(a=0.72333566*AU, 
              e=0.00677672, 
              inc=3.39467605,
              raan= 76.67984255,
              aop=131.60246718,
              mu = SUN_MU)


transfer_short = Orbit(mu=SUN_MU)
transfer_long = Orbit(mu=SUN_MU)


"""
Using JPL data to get postion and velocity of earth and mars @ Depature
"""
solar_system_ephemeris.set('de432s')  # Ephemeris from 1950 - 2050
depature_date = Time("2026-11-08")
tof = TimeDelta(100, format='jd')
arrival_date = depature_date + tof
print(f'{arrival_date}\n')

# ECI since all motion are heliocentric and the barycentric frame is centered at the sun - moves at constant velocity
r1_earth_eci, v1_earth_eci = get_body_barycentric_posvel('earth', depature_date)
r1_mars_eci, v1_mars_eci = get_body_barycentric_posvel('mars', depature_date)
r1_venus_eci,v1_venus_eci = get_body_barycentric_posvel('venus',depature_date)


"""
Need to get sun position and velocity to transform
earth and mars baycentric cords to helio-centric
"""
# Position of Sun
r_sun1, v_sun1 = get_body_barycentric_posvel('sun', depature_date)

# Position & Velocity of earth respect to sun @ Depature
r1_earth = (r1_earth_eci.xyz - r_sun1.xyz).to(u.m).value  # type:ignore
v1_earth = (v1_earth_eci.xyz - v_sun1.xyz).to(u.m/u.s).value  # type:ignore

# Position & Velocity of mars respect to sun @ Depature
r1_mars = (r1_mars_eci.xyz - r_sun1.xyz).to(u.m).value  # type:ignore
v1_mars = (v1_mars_eci.xyz - v_sun1.xyz).to(u.m/u.s).value  # type:ignore

# Position & Velocity of venus respect to sun @ Depature
r1_venus = (r1_venus_eci.xyz - r_sun1.xyz).to(u.m).value  # type:ignore
v1_venus = (v1_venus_eci.xyz - v_sun1.xyz).to(u.m/u.s).value  # type:ignore

# Propogate using RK4

def RK4_single_step(fun, dt, t0, y0, fun_arg: list):
    k1 = fun(t0, y0, fun_arg)
    k2 = fun((t0 + (dt/2)), (y0 + ((dt/2)*k1)), fun_arg)
    k3 = fun((t0 + (dt/2)), (y0 + ((dt/2)*k2)), fun_arg)
    k4 = fun((t0 + dt), (y0 + (dt*k3)), fun_arg)
    y1 = y0 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y1

def y_dot(t, y, fun_arg):
    mu = fun_arg[0]
    rx, ry, rz, vx, vy, vz = y  # Deconstruct State to get r_vec
    # print(t)
    r = np.array([rx, ry, rz])
    r_norm = np.linalg.norm(r)
    ax, ay, az = -r*mu/r_norm**3  # Two body Problem ODE
    return np.array([vx, vy, vz, ax, ay, az])

dt = 86400

# made this function to make it easier to propogate several orbits
def propagate_rk4(r0, v0, mu, tspan, dt):
    n_steps = int((tspan)/dt)
    ys = np.zeros((n_steps, 6))
    ts = np.zeros((n_steps, 1))
    y0 = np.concatenate((r0, v0))
    ys[0] = y0
    ts[0] = 0
    fun_arg = [mu]

    for i in range(n_steps - 1):
        ts[i+1] = ts[i] + dt
        ys[i+1] = RK4_single_step(y_dot, dt, ts[i], ys[i], fun_arg=fun_arg)
        
    rs = ys[:, :3]
    vs = ys[:, 3:6]

    return (rs, vs)


# Propogate orbits of Earth and Mars during TOF 
earth_rs, earth_vs = propagate_rk4(r1_earth, v1_earth, earth.mu, tspan=(tof.sec), dt=dt)
mars_rs, mars_vs = propagate_rk4(r1_mars, v1_mars, mars.mu, tspan=(tof.sec), dt=dt)

# Final Position and velocity of Mars
r2_mars = mars_rs[-1]
v2_mars = mars_vs[-1]

"""
Solving for lamberts.  
"""

# (v1_short, v2_short),= izzo.lambert(k, r1_earth*u.m, r2_mars*u.m , (tof.sec*u.s))
print("VALLADO FUNCTION")
(v1_short, v2_short),= vallado.lambert(k, r1_earth*u.m, r2_mars*u.m , (tof.sec*u.s),short=True)
print("Short Orbit Transfer")
print(f"Departure velocity: {v1_short}")
print(f"Arrival velocity: {v2_short}\n")

# (v1_long, v2_long),= vallado.lambert(k, r1_earth*u.m, r2_mars*u.m , (tof.sec*u.s),short=False,numiter=100)
# print("Long Orbit Transfer")
# print(f"Departure velocity: {v1_long}")
# print(f"Arrival velocity: {v2_long}\n")

print("JONS FUNCTION")
print("Short Orbit Transfer")
transfer_short.a, transfer_short.p, transfer_short.e, transfer_short_v1, transfer_short_v2 = lambert_solver(r1_earth, r2_mars, (tof.sec), transfer_short.mu, desired_path='short')
print(f'Short Transfer semi major axis is {transfer_short.a/1000} km -->  {(transfer_short.a/1000/149597870.7)} AU ')
print(f'Short Transfer Eccentricity is: {transfer_short.e}')
print(f'Departure velocity: {transfer_short_v1/1000} km/s')
print(f'Arrival velocity: {transfer_short_v2/1000} km/s\n')

print("Long Orbit Transfer")
transfer_long.a, transfer_long.p, transfer_long.e, transfer_long_v1, transfer_long_v2 = lambert_solver(r1_earth, r2_mars, (tof.sec), transfer_short.mu, desired_path='long')
print(f'Long Transfer semi major axis is {transfer_long.a/1000} km --> {(transfer_long.a/1000/149597870.7)} AU')
print(f'Long Transfer Eccentricity is: {transfer_long.e}')
print(f'Departure velocity: {transfer_long_v1/1000} km/s')
print(f'Arrival velocity: {transfer_long_v2/1000} km/s\n')

print("VRAJ FUNCTION")
print("Short Orbit Transfer")
transfer_short.a, transfer_short.p, transfer_short.e, transfer_short_v1, transfer_short_v2 = universal_lambert( r1_earth, r2_mars, (tof.sec), transfer_short.mu, desired_path='short')
print(f'Short Transfer semi major axis is {transfer_short.a/1000} km -->  {(transfer_short.a/1000/149597870.7)} AU ')
print(f'Short Transfer Eccentricity is: {transfer_short.e}')
print(f'Departure velocity: {transfer_short_v1/1000} km/s')
print(f'Arrival velocity: {transfer_short_v2/1000} km/s\n')


print("Long Orbit Transfer")
transfer_long.a, transfer_long.p, transfer_long.e, transfer_long_v1, transfer_long_v2 = universal_lambert(r1_earth, r2_mars, (tof.sec), transfer_long.mu, desired_path='long')
print(f'Long Transfer semi major axis is {transfer_long.a/1000} km --> {(transfer_long.a/1000/149597870.7)} AU')
print(f'Long Transfer Eccentricity is: {transfer_long.e}')
print(f'Departure velocity: {transfer_long_v1/1000} km/s')
print(f'Arrival velocity: {transfer_long_v2/1000} km/s')


"""
Propogate transfer orbits
"""
transfer_r1 = r1_earth
transfer_short_rs, transfer_short_vs = propagate_rk4( transfer_r1, transfer_short_v1, transfer_short.mu, tspan=tof.sec, dt=dt)
transfer_long_rs, transfer_long_vs = propagate_rk4( transfer_r1, transfer_long_v1, transfer_long.mu, tspan=tof.sec, dt=dt)

"""
Calcs
"""

dv1_short = transfer_short_v1 - v1_earth
dv2_short = v2_mars - transfer_short_v2
dv_short = np.linalg.norm(dv1_short) + np.linalg.norm(dv2_short)

dv1_long = transfer_long_v1 - v1_earth
dv2_long = v2_mars - transfer_long_v2
dv_long = np.linalg.norm(dv1_long) + np.linalg.norm(dv2_long)

 # Full Propogated Orbits: 1 Period
earth_full, earth_full = propagate_rk4(r1_earth, v1_earth, earth.mu, tspan=(earth.period(earth.a,earth.mu)), dt=dt)
mars_full, mars_full = propagate_rk4(r1_mars, v1_mars, mars.mu, tspan=(mars.period(mars.a,mars.mu)), dt=dt)
plot = True
if plot == True:
    """- - - - - - - - - - - - - - - -PLOTTING- - - - - - - - - - - - - - - -"""
   
    fig = plt.figure()
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(earth_full[:, 0], earth_full[:, 1], earth_full[:, 2], color='green', label='earth')
    ax.plot(mars_full[:, 0], mars_full[:, 1], mars_full[:, 2], color='red', label='mars')
    ax.plot(transfer_short_rs[:, 0], transfer_short_rs[:, 1], transfer_short_rs[:, 2], color='orange', label='short transfer', linestyle='--')
    ax.plot(transfer_long_rs[:, 0], transfer_long_rs[:, 1],transfer_long_rs[:, 2], color='blue', label='long transfer', linestyle='--')

    # Add Sun
    ax.scatter(0, 0, 0, color='yellow', s=15, marker='o', edgecolor='k', label="SUN")

    # Add Earth Departure Point
    ax.scatter(earth_rs[0, 0], earth_rs[0, 1], earth_rs[0, 2],color='green', s=15, marker='o', edgecolor='k', label="Earth Depature")

    # Add Earth Arrival Point
    ax.scatter(earth_rs[-1, 0], earth_rs[-1, 1], earth_rs[-1, 2],color='green', s=15, marker='o', edgecolor='k', label="Earth Arrival")

    # Add Mars Departure Point
    ax.scatter(mars_rs[0, 0], mars_rs[0, 1], mars_rs[0, 2],color='red', s=15, marker='o', edgecolor='k', label="Mars Depature")

    # Add Mars Arrival Point
    ax.scatter(mars_rs[-1, 0], mars_rs[-1, 1], mars_rs[-1, 2],color='red', s=15, marker='o', edgecolor='k', label="Mars Arrival")

    # formatting
    ax.set_title(
        f"Earth–Mars Transfer Orbits {depature_date.strftime('%Y-%m-%d')} - {arrival_date.strftime('%Y-%m-%d')}", fontsize=14, pad=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc='right')
    ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))

    plt.show()


# """
# Print Summary
# """
# print(f'---------------------------------------------')
# print(f'------------------ SUMMARY ------------------')
# print(f'---------------------------------------------')

# print(f'------------------ @ Depature: {depature_date} ------------------')
# print(f'earth_r1= {np.linalg.norm(r1_earth)/1000}')
# print(f'earth_v1= {np.linalg.norm(v1_earth)/1000}')
# print(f'mars_r1= {np.linalg.norm(r1_mars)/1000}')
# print(f'mars_v1= {np.linalg.norm(v1_mars)/1000}')


# print(f'------------------DelatV SHORT------------------')
# print(f'dv1_short= {dv1_short/1000}')
# print(f'dv1_short_MAG= {np.linalg.norm(dv1_short)/1000}')
# print(f'dv2_short= {dv2_short/1000}')
# print(f'dv2_short_MAG= {np.linalg.norm(dv2_short)/1000}')
# print(f'dv_short= {dv2_short/1000}')
# print(f'dv_short_MAG= {dv_short/1000}')


# print(f'------------------DelatV LONG------------------')
# print(f'dv1_long= {dv1_long/1000}')
# print(f'dv1_long_MAG= {np.linalg.norm(dv1_long)/1000}')
# print(f'dv2_long= {dv2_long/1000}')
# print(f'dv2_long_MAG= {np.linalg.norm(dv2_long)/1000}')
# print(f'dv_long= {dv_long/1000}')
# print(f'dv_long_MAG= {dv_long/1000}')


# """
# print(f'------------------LAMBERTS SHORT------------------')
# print(f'transfer_v1_MAG= {np.linalg.norm(transfer_short_v1)/1000}')
# print(f'transfer_v1= {(transfer_short_v1)/1000}')
# print(f'transfer_v2_MAG= {np.linalg.norm(transfer_short_v2)/1000}')
# print(f'transfer_v2= {(transfer_short_v2)/1000}')
# print(f'Detal V = {dv_short/1000}')  # type:ignore
# print(f'------------------LAMBERTS LONG------------------')
# print(f'transfer_v1= {np.linalg.norm(transfer_long_v1)/1000}')
# print(f'transfer_v2= {np.linalg.norm(transfer_long_v2)/1000}')
# print(f'Detal V = {dv_long/1000}')  # type:ignore
# """
