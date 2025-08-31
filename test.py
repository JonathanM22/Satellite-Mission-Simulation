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
        self.f0 = f0
        self.inc = inc
        self.raan = raan
        self.aop = aop
        self.mu = mu
        self.energy = None
        self.p = None
        self.h = None

    # Function to calcualte r based on f


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


test = Orbit(a=26564*1000,
             e=0.7411,
             f0=np.deg2rad(30),
             inc=np.deg2rad(0),
             raan=np.deg2rad(200),
             aop=np.deg2rad(-90),
             mu=earth_mu)

test2 = Orbit(a=26564*1000,
              e=0.7411,
              f0=np.deg2rad(30),
              inc=np.deg2rad(45),
              raan=np.deg2rad(200),
              aop=np.deg2rad(-90),
              mu=earth_mu)
"""- - - - - - - - - - - - - - - -Test Orbit 1- - - - - - - - - - - - - - - -"""
test.p = (test.a*(1-test.e**2))  # type: ignore
test_r1 = (test.p) / (1 + test.e*np.cos(test.f0))

r_pqw, v_pqw = orbelm_2_pqw(test_r1, test.f0,
                            test.e, test.p, test.mu)

test_r0, test_v0 = perif_2_eci(r_pqw, v_pqw, test.inc,
                               test.raan, test.aop)

# Intitalize Arrays for Solver
tspan = (2*np.pi)*np.sqrt(test.a**3/test.mu)
print(tspan)
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

"""- - - - - - - - - - - - - - - -Test Orbit 2- - - - - - - - - - - - - - - -"""
test2.p = (test2.a*(1-test2.e**2))  # type: ignore
test2_r1 = (test2.p) / (1 + test2.e*np.cos(test2.f0))

r_pqw, v_pqw = orbelm_2_pqw(test2_r1, test2.f0,
                            test2.e, test2.p, test2.mu)

test2_r0, test2_v0 = perif_2_eci(r_pqw, v_pqw, test2.inc,
                                 test2.raan, test2.aop)

# Intitalize Arrays for Solver
tspan = (2*np.pi)*np.sqrt(test2.a**3/test2.mu)
print(tspan)
dt = 100
n_steps = int(np.ceil(tspan/dt))
ys = np.zeros((n_steps, 6))
ts = np.zeros((n_steps, 1))

# Intial condition of solver
test2_y0 = np.concatenate((test2_r0, test2_v0))
ys[0] = test2_y0
step = 1

# Intiate Solver
solver = ode(y_dot)
solver.set_integrator('lsoda')
solver.set_initial_value(test2_y0, 0)
solver.set_f_params(earth_mu)


while solver.successful and step < n_steps:
    solver.integrate(solver.t+dt)
    ts[step] = solver.t
    ys[step] = solver.y
    step += 1

rs2 = ys[:, :3]


"""- - - - - - - - - - - - - - - -PLOTTING- - - - - - - - - - - - - - - -"""
# fig = plt.figure(figsize(18, 6))
ax = plt.figure().add_subplot(projection='3d')
ax.plot(rs[:, 0], rs[:, 1], rs[:, 2])
ax.plot(rs2[:, 0], rs2[:, 1], rs2[:, 2])
ax.set_aspect('equal')
plt.show()

print(rs1[:5])
print('/n')
print(rs2[:5])
