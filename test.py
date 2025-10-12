"""
main2.py Uses JPL ephemeris data as r1 and propagates r2 to calculate the the lamberst problem 
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

transfer = Orbit(mu=SUN_MU)

# Set Up r1 & r2 from jpl ephemeris
solar_system_ephemeris.set('de432s')  # Ephemeris from 1950 - 2050
depature_date = Time("2020-06-01")
# tof = TimeDelta(365, format='jd')
# arrival_date = depature_date + tof
arrival_date = Time("2021-01-01")

tof = (arrival_date - depature_date).sec

r1_eph, v1_eph = get_body_barycentric_posvel('earth', depature_date)
r1_mars, v1_mars = get_body_barycentric_posvel('mars', depature_date)

"""
Need to get sun position and velocity to transform
baycentric cords to helio-centric
"""
# Position of Sun
r_sun1, v_sun1 = get_body_barycentric_posvel('sun', depature_date)

# Position & Velocity of earth respect to sun
r1 = (r1_eph.xyz - r_sun1.xyz).to(u.m).value  # type:ignore
v1 = (v1_eph.xyz - v_sun1.xyz).to(u.m/u.s).value  # type:ignore

# Position & Velocity of mars depature respect to sun
r1_mars = (r1_mars.xyz - r_sun1.xyz).to(u.m).value  # type:ignore
v1_mars = (v1_mars.xyz - v_sun1.xyz).to(u.m/u.s).value  # type:ignore

"""
Propogate from JPL data to get the arrivial position of the bodies. 
"""
dt = 86400
earth_rs, earth_vs = propogate_orbit(
    r1, v1, earth.mu, tspan=tof, dt=dt)
mars_rs, mars_vs = propogate_orbit(
    r1_mars, v1_mars, mars.mu, tspan=tof, dt=dt)

r2_mars = mars_rs[-1]
v2_mars = mars_vs[-1]


# Solve for transfer orbit via lamberts
transfer.a, transfer.p, transfer.e, transfer_v1, transfer_v2 = lambert_solver(
    r1, r2_mars, (tof), transfer.mu, desired_path='short')  # type:ignore

transfer_r1 = r1

print("-------------------------")
print(transfer_v1)
print("-------------------------")

plot = False
if plot == True:
    dt = 86400
    """- - - - - - - - - - - - - - - -PLOTTING- - - - - - - - - - - - - - - -"""
    transfer_rs, transfer_vs = propogate_orbit(
        transfer_r1, transfer_v1, transfer.mu, tspan=tof, dt=dt)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(earth_rs[:, 0], earth_rs[:, 1],
            earth_rs[:, 2], color='green', label='earth')
    ax.plot(mars_rs[:, 0], mars_rs[:, 1],
            mars_rs[:, 2], color='red', label='mars')
    ax.plot(transfer_rs[:, 0], transfer_rs[:, 1],
            transfer_rs[:, 2], color='orange', label='transfer')

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

dv1 = transfer_v1-v1
dv2 = v2_mars-transfer_v2
dv = np.linalg.norm(dv1) + np.linalg.norm(dv2)

print(f'transfer_v1= {np.linalg.norm(transfer_v1)/1000}')
print(f'transfer_v2= {np.linalg.norm(transfer_v2)/1000}')
print(f'earth_v1= {np.linalg.norm(v1)/1000}')
print(f'mars_v2= {np.linalg.norm(v2_mars)/1000}')
print(f'c3 = {(np.linalg.norm(transfer_v1)-np.linalg.norm(v1))/1000} km/s')
print(
    f'V_inf_arrvial = {(np.linalg.norm(v2_mars)-np.linalg.norm(transfer_v2))/1000} km/s')

print(f'Detal V1 = {np.linalg.norm(dv1)/1000}')
print(f'Detal V2 = {np.linalg.norm(dv2)/1000}')
print(f'Detal VT = {dv/1000}')
