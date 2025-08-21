import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode


# Constants
# ALL constants are in SI UNITS! (meters, seconds, etc.)
earth_rad = 6.371 * 10**6
au = 149597870.691
sun_mu = 1.327 * 10**20
earth_mu = 3.956 * 10**14


class Orbit:
    def __init__(self, mu, a=None, e=None, f0=None, inc=None, raan=None, aop=None):
        self.a = a
        self.e = e
        try:
            self.f0 = np.deg2rad(f0)  # type: ignore
            self.inc = np.deg2rad(inc)  # type: ignore
            self.raan = np.deg2rad(raan)  # type: ignore
            self.aop = np.deg2rad(aop)  # type: ignore
        except:
            pass
        self.mu = mu
        self.energy = None
        self.p = None
        self.h = None
        self.e_hat = None

    def propogate(self, r, v, tspan, dt):
        n_steps = int(np.ceil(tspan/dt))
        ys = np.zeros((n_steps, 6))
        ts = np.zeros((n_steps, 1))

        # Intial condition of solver
        y0 = np.concatenate((r, v))
        ys[0] = y0
        step = 1

        # Intiate Solver
        solver = ode(y_dot)
        solver.set_integrator('lsoda')
        solver.set_initial_value(y0, 0)
        solver.set_f_params(self.mu)

        while solver.successful and step < n_steps:
            solver.integrate(solver.t+dt)
            ts[step] = solver.t
            ys[step] = solver.y
            step += 1

        rs = ys[:, :3]
        vs = ys[:4, :]

        return (rs, vs)


def orbelm_2_pqw(r, f, e, p, mu):

    r_pqw = np.array([r*np.cos(f), r*np.sin(f), 0])
    v_pqw = np.array(
        [-np.sqrt(mu/p)*np.sin(f), np.sqrt(mu/p)*(e + np.cos(f)), 0])

    return r_pqw, v_pqw


def perif_2_eci(r_pqw, v_pqw, inc, raan, aop):
    # Rotation matrices
    R1 = np.array([  # Third axis rotation about raan
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan),  np.cos(raan), 0],
        [0,             0,            1]
    ])
    R2 = np.array([  # First axis rotation about inc
        [1, 0,              0],
        [0, np.cos(inc), -np.sin(inc)],
        [0, np.sin(inc),  np.cos(inc)]
    ])
    R3 = np.array([  # Third axis rotation about aop
        [np.cos(aop), -np.sin(aop), 0],
        [np.sin(aop),  np.cos(aop), 0],
        [0,            0,           1]
    ])
    perif_2_eci_DCM = R1 @ R2 @ R3
    r_eci = perif_2_eci_DCM @ r_pqw
    v_eci = perif_2_eci_DCM @ v_pqw

    return r_eci, v_eci


def y_dot(t, y, mu):
    # print(y)
    rx, ry, rz, vx, vy, vz = y  # Deconstruct State to get r_vec
    r = np.array([rx, ry, rz])
    r_norm = np.linalg.norm(r)
    ax, ay, az = -r*mu/r_norm**3  # Two body Problem ODE
    return [vx, vy, vz, ax, ay, az]


# Input degrees
parking = Orbit(a=au,
                e=0.174,
                f0=0,
                inc=1,
                raan=200,
                aop=37,
                mu=sun_mu)

target = Orbit(a=1.54*au,
               e=0.4,
               f0=0,
               inc=50,
               raan=40,
               aop=25,
               mu=sun_mu)

transfer = Orbit(mu=sun_mu)

# Calculate r0 & v0 off f0
parking.p = (parking.a*(1-parking.e**2))  # type: ignore
parking_r0 = (parking.p) / (1 + parking.e*np.cos(parking.f0))
parking_r0_pqw, parking_v0_pqw = orbelm_2_pqw(
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
parking_r1_pqw, parking_v1_pqw = orbelm_2_pqw(
    parking_r1, parking_f1, parking.e, parking.p, parking.mu)
parking_r1_eci, parking_v1_eci = perif_2_eci(
    parking_r1_pqw, parking_v1_pqw, parking.inc, parking.raan, parking.aop)
print(f'parking r1 = {parking_r1_eci}')
# print(f'parking v1 = {parking_v1_eci}')


# Calculate r0 & v0 off f0
target.p = (target.a*(1-target.e**2))  # type: ignore
target_r0 = (target.p) / (1 + target.e*np.cos(target.f0))
target_r0_pqw, target_v0_pqw = orbelm_2_pqw(
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
target_r1_pqw, target_v1_pqw = orbelm_2_pqw(
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
# -----
target.p = (target.a*(1-target.e**2))  # type: ignore
target.f0 = np.deg2rad(90)-target.aop
# Forcing r2 to be on line of nodes +/- 90deg
target_r2 = (target.p) / (1 + target.e*np.cos(target.f0))

# Intial r1 & r2 in ECI frames
parking_r1_pqw, parking_v1_pqw = orbelm_2_pqw(
    parking_r1, parking.f0, parking.e, parking.p, parking.mu)
parking_r1_eci, parking_v1_eci = perif_2_eci(
    parking_r1_pqw, parking_v1_pqw, target.inc, target.raan, target.aop)

target_r2_pqw, target_v2_pqw = orbelm_2_pqw(
    target_r2, target.f0, target.e, target.p, target.mu)
target_r2_eci, target_v2_eci = perif_2_eci(
    target_r2_pqw, target_v2_pqw, target.inc, target.raan, target.aop)

# Check for hohmann
print(f"r1 = {parking_r1_eci}")
print(f"r2 = {target_r2_eci}")
"""

# -----------------


"""
test.p = (test.a*(1-test.e**2))  # type: ignore
test_r1 = (test.p) / (1 + test.e*np.cos(test.f0))

r_pqw, v_pqw = orbelm_2_pqw(test_r1, test.f0,
                            test.e, test.p, test.mu)

test_r0, test_v0 = perif_2_eci(r_pqw, v_pqw, test.inc,
                               test.raan, test.aop)

# Intitalize Arrays for Solver
tspan = (2*np.pi)*np.sqrt(test.a**3/test.mu)
# print(tspan)
dt = 100
n_steps = int(np.ceil(tspan/dt))
ys = np.zeros((n_steps, 6))
ts = np.zeros((n_steps, 1))

# Intial condition of solver
test_y0 = np.concatenate((test_r0, test_v0))
ys[0] = test_y0
step = 1

# Intiate Solver
solver = ode(y_dot)
solver.set_integrator('lsoda')
solver.set_initial_value(test_y0, 0)
solver.set_f_params(earth_mu)


while solver.successful and step < n_steps:
    solver.integrate(solver.t+dt)
    ts[step] = solver.t
    ys[step] = solver.y
    step += 1

rs = ys[:, :3]

# print(rs)

# fig = plt.figure(figsize(18, 6))
ax = plt.figure().add_subplot(projection='3d')
ax.plot(rs[:, 0], rs[:, 1], rs[:, 2])
plt.show()
"""
