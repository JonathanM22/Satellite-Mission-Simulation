from orbit import *
import numpy as np
from scipy.integrate import ode
from scipy import optimize
import matplotlib.pyplot as plt

# Constants
# ALL constants are in SI UNITS! (meters, seconds, etc.)
# Also for formattng constants are ALL_CAPS
EARTH_RAD = 6.371 * 10**6
AU = 1.496 * 10**11
SUN_MU = 1.327 * 10**20


# Use Table 1 https://ssd.jpl.nasa.gov/planets/approx_pos.html
earth = Orbit(a=AU,
              e=0.0167,
              inc=0,
              raan=0,
              aop=102.9,
              mu=SUN_MU)

mars = Orbit(a=1.523*AU,
             e=0.093,
             inc=1.849,
             raan=49.5,
             aop=-23.9,
             mu=SUN_MU)

transfer = Orbit(mu=SUN_MU)

# Set Up r1 and r2 & TOF
earth_f1 = 90
mars_f1 = 0
TOF = 365*86400

r1 = earth.r_at_true_anomaly(earth_f1)
earth.p = earth.calc_p()
r1_pqw, v1_pqw = orb_2_pqw(r1, earth_f1, earth.e, earth.p, earth.mu)
r1_eci, v1_eci = perif_2_eci(r1_pqw, v1_pqw, earth.inc, earth.raan, earth.aop)

r2 = mars.r_at_true_anomaly(mars_f1)
mars.p = mars.calc_p()
r2_pqw, v2_pqw = orb_2_pqw(r2, mars_f1, mars.e, mars.p, mars.mu)
r2_eci, v2_eci = perif_2_eci(r2_pqw, v2_pqw, mars.inc, mars.raan, mars.aop)

# Solve for transfer orbit via lamberts
transfer.a, transfer.p, transfer.e, transfer_v1, transfer_v2 = lambert_solver(
    r1_eci, r2_eci, TOF, transfer.mu)  # type:ignore

transfer_r1 = r1_eci


plot = True
if plot == True:
    """- - - - - - - - - - - - - - - -PLOTTING- - - - - - - - - - - - - - - -"""
    earth_rs, earth_vs = propogate_orbit(
        r1_eci, v1_eci, earth.mu, tspan=earth.period(), dt=86400)
    mars_rs, mars_vs = propogate_orbit(
        r2_eci, v2_eci, mars.mu, tspan=mars.period(), dt=86400)
    transfer_rs, transfer_vs = propogate_orbit(
        transfer_r1, transfer_v1, transfer.mu, tspan=TOF, dt=86400)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(earth_rs[:, 0], earth_rs[:, 1], earth_rs[:, 2], color='green')
    ax.plot(mars_rs[:, 0], mars_rs[:, 1], mars_rs[:, 2], color='red')
    ax.plot(transfer_rs[:, 0], transfer_rs[:, 1],
            transfer_rs[:, 2], color='orange')

    # Add Sun
    ax.scatter(0, 0, 0,
               color='yellow', s=25, marker='o', edgecolor='k', label="SUN")

    # formatting
    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    plt.show()


"""
# Calculate r0 & v0 off f0
parking.p = (parking.a*(1-parking.e**2))  # type: ignore
parking_r0 = (parking.p) / (1 + parking.e*np.cos(parking.f0))
parking_r0_pqw, parking_v0_pqw = orb_2_pqw(
    parking_r0, parking.f0, parking.e, parking.p, parking.mu)
parking_r0_eci, parking_v0_eci = perif_2_eci(
    parking_r0_pqw, parking_v0_pqw, parking.inc, parking.raan, parking.aop)
print(f'parking r0 = {parking_r0_eci}')
# print(f'parking v0 = {parking_v0_eci}')

# Calculate r1 at line of nodes
parking_f1 = np.deg2rad(360) - parking.aop
# parking_f1 = 90 + parking.aop + parking.raan
# print(np.rad2deg(parking_f1))
parking_r1 = (parking.p) / (1 + parking.e*np.cos(parking_f1))
parking_r1_pqw, parking_v1_pqw = orb_2_pqw(
    parking_r1, parking_f1, parking.e, parking.p, parking.mu)
parking_r1_eci, parking_v1_eci = perif_2_eci(
    parking_r1_pqw, parking_v1_pqw, parking.inc, parking.raan, parking.aop)
print(f'parking r1 = {parking_r1_eci}')
# print(f'parking v1 = {parking_v1_eci}')


# Calculate r0 & v0 off f0
target.p = (target.a*(1-target.e**2))  # type: ignore
target_r0 = (target.p) / (1 + target.e*np.cos(target.f0))
target_r0_pqw, target_v0_pqw = orb_2_pqw(
    target_r0, target.f0, target.e, target.p, target.mu)
target_r0_eci, target_v0_eci = perif_2_eci(
    target_r0_pqw, target_v0_pqw, target.inc, target.raan, target.aop)
print(f'target r0 = {target_r0_eci}')
# print(f'parking v0 = {parking_v0_eci}')

# Calculate r2 at line of nodes
target_f1 = np.deg2rad(360) - target.aop
target_f1 = 90 + target.aop + target.raan
# print(np.rad2deg(parking_f1))
target_r1 = (target.p) / (1 + target.e*np.cos(target_f1))
target_r1_pqw, target_v1_pqw = orb_2_pqw(
    target_r1, target_f1, target.e, target.p, target.mu)
target_r1_eci, target_v1_eci = perif_2_eci(
    target_r1_pqw, target_v1_pqw, target.inc, target.raan, target.aop)
print(f'parking r1 = {target_r1_eci}')
# print(f'parking v1 = {parking_v1_eci}')

parking_r1_eci_norm = np.linalg.norm(parking_r1_eci)
target_r1_eci_norm = np.linalg.norm(target_r1_eci)

deltaf = np.acos((np.dot(parking_r1_eci, target_r1_eci)) /
                 (parking_r1_eci_norm*target_r1_eci_norm))
print(np.rad2deg(deltaf))

# Intitalize Arrays for Solver
tspan1 = (2*np.pi)*np.sqrt(parking.a**3/parking.mu)  # type: ignore
dt = 10

parking_rs, parking_vs = parking.propogate(
    parking_r0_eci, parking_v0_eci, tspan1, dt)

tspan2 = (2*np.pi)*np.sqrt(target.a**3/target.mu)  # type: ignore
target_rs, target_vs = target.propogate(
    target_r0_eci, target_v0_eci, tspan2, dt)

# fig = plt.figure(figsize(18, 6))
ax = plt.figure().add_subplot(projection='3d')
ax.plot(parking_rs[:, 0], parking_rs[:, 1], parking_rs[:, 2], color='green')
ax.plot(target_rs[:, 0], target_rs[:, 1], target_rs[:, 2], color='red')

# add the Sun at the origin
ax.scatter(parking_r1_eci[0], parking_r1_eci[1], parking_r1_eci[1], color='green', s=50,
           marker='o', edgecolor='k', label="pf1")

ax.scatter(0, 0, 0, color='yellow', s=50,
           marker='o', edgecolor='k', label="tf1")

# formatting
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.legend()
ax.set_box_aspect([1, 1, 1])  # keep equal aspect ratio

plt.show()

plt.show()

"""
