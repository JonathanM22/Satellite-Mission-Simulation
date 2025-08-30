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
EARTH_MU = 3.956 * 10**144


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
earth_f1 = 0
mars_f1 = 0
TOF = 150*86400

r1 = earth.r_at_true_anomaly(earth_f1)
earth.p = earth.calc_p()
r1_pqw, v1_pqw = orb_2_pqw(r1, earth_f1, earth.e, earth.p, earth.mu)
r1_eci, v1_eci = perif_2_eci(r1_pqw, v1_pqw, earth.inc, earth.raan, earth.aop)

r2 = mars.r_at_true_anomaly(mars_f1)
mars.p = mars.calc_p()
r2_pqw, v2_pqw = orb_2_pqw(r1, mars_f1, mars.e, mars.p, mars.mu)
r2_eci, v2_eci = perif_2_eci(r2_pqw, v2_pqw, mars.inc, mars.raan, mars.aop)

delta_f = np.arccos((np.dot(r1_eci, r2_eci)) / (r1*r2))
# delta_f = 2*np.pi - delta_f

print(f'\ndelta f: {np.rad2deg(delta_f)}\n')

"""
Override vars to mimc 458 class example "A550_03_17_25.pdf"
"""
# r1 = 1
# r2 = 0.72
# delta_f = np.deg2rad(60)
# TOF = 150/365
# transfer.mu = 4*(np.pi**2)

# Calculate t_parab
c = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(delta_f))
s = (r1 + r2 + c) / 2

if 0 <= delta_f < np.pi:
    t_parab = (1/3) * np.sqrt(2/transfer.mu) * (s**(3/2) - ((s-c)**(3/2)))
elif np.pi <= delta_f < 2*np.pi:
    t_parab = (1/3) * np.sqrt(2/transfer.mu) * (s**(3/2) + ((s-c)**(3/2)))

if TOF > t_parab:  # type:ignore
    print("elliptical solution")
else:
    print('hyperbolic solution')

# Calculate minumum transfer
a_m = s/2
alpha_m = np.pi
if 0 <= delta_f < np.pi:
    beta_m = 2*np.arcsin(np.sqrt((s-c)/s))
elif np.pi <= delta_f < 2*np.pi:
    beta_m = -2*np.arcsin(np.sqrt((s-c)/s))

tm = np.sqrt((s**3)/(8 * transfer.mu)) * \
    (np.pi - beta_m + np.sin(beta_m))  # type:ignore

# Define alpha and beta
if TOF <= tm:
    def alpha(a): return 2*np.arcsin(np.sqrt((s/(2*a))))
elif TOF > tm:
    def alpha(a): return 2*np.pi - 2*np.arcsin(np.sqrt((s/(2*a))))

if 0 <= delta_f < np.pi:
    def beta(a): return 2*np.arcsin(np.sqrt((s-c)/(2*a)))
elif np.pi <= delta_f < 2*np.pi:
    def beta(a): return -2*np.arcsin(np.sqrt((s-c)/(2*a)))

# Solve for a


def lambert_eq(a): return ((np.sqrt(a**3)) * (alpha(a) - np.sin(alpha(a)
                                                                ) - beta(a) + np.sin(beta(a)))) - ((np.sqrt(transfer.mu))*TOF)


transfer.a = optimize.brentq(lambert_eq, a_m, 5*AU)
print(f'lamber a {transfer.a}')


transfer.p = (((4*transfer.a)*(s-r1)*(s-r2))/(c**2)) * \
    (np.sin((alpha(transfer.a) + beta(transfer.a))/2)**2)  # type:ignore
transfer.e = np.sqrt(1 - (transfer.p/transfer.a))

print(5)

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
