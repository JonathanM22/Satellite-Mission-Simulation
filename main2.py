"""
main2.py Uses JPL ephemeris data as r1 and r2 to calculate the the lamberst problem 
"""
from orbit import *
import numpy as np
from scipy.integrate import ode
from scipy import optimize
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.time import TimeDelta
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel

# Constants
# ALL constants are in SI UNITS! (meters, seconds, etc.)
# Also for formattng constants are ALL_CAPS
EARTH_RAD = 6.371 * 10**6
AU = 1.496 * 10**11
SUN_MU = 1.327 * 10**20


# Use Table 1 https://ssd.jpl.nasa.gov/planets/approx_pos.html
earth = Orbit(a=1.00000261*AU,
              e=0.01671123,
              inc=-0.00001531,
              raan=0,
              aop=102.93768193,
              mu=SUN_MU)

mars = Orbit(a=1.523*AU,
             e=0.093,
             inc=1.849,
             raan=49.5,
             aop=-23.9,
             mu=SUN_MU)

transfer_short = Orbit(mu=SUN_MU)
transfer_long = Orbit(mu=SUN_MU)

"""
Using JPL data to get postion and velocity of earth and mars @ Depature
"""
solar_system_ephemeris.set('de432s')  # Ephemeris from 1950 - 2050
depature_date = Time("2026-11-08")
tof = TimeDelta(304, format='jd')
arrival_date = depature_date + tof
print(arrival_date)

r1_earth_eph, v1_earth_eph = get_body_barycentric_posvel(
    'earth', depature_date)
r1_mars_eph, v1_mars_eph = get_body_barycentric_posvel('mars', depature_date)

"""
Need to get sun position and velocity to transform
earth and mars baycentric cords to helio-centric
"""
# Position of Sun
r_sun1, v_sun1 = get_body_barycentric_posvel('sun', depature_date)

# Position & Velocity of earth respect to sun @ Depature
r1_earth = (r1_earth_eph.xyz - r_sun1.xyz).to(u.m).value  # type:ignore
v1_earth = (v1_earth_eph.xyz - v_sun1.xyz).to(u.m/u.s).value  # type:ignore

# Position & Velocity of mars respect to sun @ Depature
r1_mars = (r1_mars_eph.xyz - r_sun1.xyz).to(u.m).value  # type:ignore
v1_mars = (v1_mars_eph.xyz - v_sun1.xyz).to(u.m/u.s).value  # type:ignore

"""
Propogate from JPL data to get the arrivial position of the bodies. 
"""
dt = 86400
earth_rs, earth_vs = propogate_orbit(
    r1_earth, v1_earth, earth.mu, tspan=tof.sec, dt=dt)
mars_rs, mars_vs = propogate_orbit(
    r1_mars, v1_mars, mars.mu, tspan=tof.sec, dt=dt)

r2_mars = mars_rs[-1]
v2_mars = mars_vs[-1]

"""
Solving for lamberts.  
"""
transfer_short.a, transfer_short.p, transfer_short.e, transfer_short_v1, transfer_short_v2 = lambert_solver(
    # type:ignore
    r1_earth, r2_mars, (tof.sec), transfer_short.mu, desired_path='short')

transfer_long.a, transfer_long.p, transfer_long.e, transfer_long_v1, transfer_long_v2 = lambert_solver(
    # type:ignore
    r1_earth, r2_mars, (tof.sec), transfer_short.mu, desired_path='long')


transfer_r1 = r1_earth

plot = True
if plot == True:
    dt = 86400
    """- - - - - - - - - - - - - - - -PLOTTING- - - - - - - - - - - - - - - -"""
    transfer_short_rs, transfer_short_vs = propogate_orbit(
        transfer_r1, transfer_short_v1, transfer_short.mu, tspan=tof.sec, dt=dt)
    transfer_long_rs, transfer_long_vs = propogate_orbit(
        transfer_r1, transfer_long_v1, transfer_long.mu, tspan=tof.sec, dt=dt)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(earth_rs[:, 0], earth_rs[:, 1],
            earth_rs[:, 2], color='green', label='earth')
    ax.plot(mars_rs[:, 0], mars_rs[:, 1],
            mars_rs[:, 2], color='red', label='mars')
    ax.plot(transfer_short_rs[:, 0], transfer_short_rs[:, 1],
            transfer_short_rs[:, 2], color='orange', label='short transfer', linestyle='--')
    ax.plot(transfer_long_rs[:, 0], transfer_long_rs[:, 1],
            transfer_long_rs[:, 2], color='blue', label='long transfer', linestyle='--')

    # Add Sun
    ax.scatter(0, 0, 0,
               color='yellow', s=15, marker='o', edgecolor='k', label="SUN")

    # Add Earth Departure Point
    ax.scatter(earth_rs[0, 0], earth_rs[0, 1], earth_rs[0, 2],
               color='green', s=15, marker='o', edgecolor='k', label="Earth Depature")

    # Add Earth Arrival Point
    ax.scatter(earth_rs[-1, 0], earth_rs[-1, 1], earth_rs[-1, 2],
               color='green', s=15, marker='o', edgecolor='k', label="Earth Arrival")

    # Add Mars Departure Point
    ax.scatter(mars_rs[0, 0], mars_rs[0, 1], mars_rs[0, 2],
               color='red', s=15, marker='o', edgecolor='k', label="Mars Depature")

    # Add Mars Arrival Point
    ax.scatter(mars_rs[-1, 0], mars_rs[-1, 1], mars_rs[-1, 2],
               color='red', s=15, marker='o', edgecolor='k', label="Mars Arrival")

    # formatting
    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc='right')
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.show()

    dv1_short = np.linalg.norm(transfer_short_v1 - v1_earth)
    dv2_short = np.linalg.norm(v2_mars - transfer_short_v2)
    dv_short = dv1_short + dv2_short

    dv1_long = np.linalg.norm(transfer_long_v1 - v1_earth)
    dv2_long = np.linalg.norm(v2_mars - transfer_long_v2)
    dv_long = dv1_long + dv2_long

print(f'earth_v1= {np.linalg.norm(v1_mars)/1000}')
print(f'mars_v2= {np.linalg.norm(v2_mars)/1000}')
print(f'------------------LAMBERTS SHORT------------------')
print(f'transfer_v1= {np.linalg.norm(transfer_short_v1)/1000}')
print(f'transfer_v2= {np.linalg.norm(transfer_short_v2)/1000}')
print(f'Detal V = {dv_short/1000}')
print(f'------------------LAMBERTS LONG------------------')
print(f'transfer_v1= {np.linalg.norm(transfer_long_v1)/1000}')
print(f'transfer_v2= {np.linalg.norm(transfer_long_v2)/1000}')
print(f'Detal V = {dv_long/1000}')
