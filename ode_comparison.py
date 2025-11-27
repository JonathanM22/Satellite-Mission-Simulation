from orbit import *
from Orbit_util import *
import numpy as np
from scipy.integrate import ode
from scipy import optimize
import matplotlib.pyplot as plt
import time

# Constants
# ALL constants are in SI UNITS! (meters, seconds, etc.)
# Also for formattng constants are ALL_CAPS
EARTH_RAD = 6.371 * 10**6
AU = 1.496 * 10**11
SUN_MU = 1.327 * 10**20

# Use Table 1 https://ssd.jpl.nasa.gov/planets/approx_pos.html
earth = Orbit(a=AU,
              e=0.0167,
              inc=np.deg2rad(25),
              raan=np.deg2rad(0),
              aop=np.deg2rad(102.9),
              mu=SUN_MU)
earth.p = earth.calc_p(earth.a, earth.e)

transfer = Orbit(mu=SUN_MU)

# Set Up r1 and r2 & TOF
earth_f1 = np.deg2rad(90)


r1 = earth.r_at_true_anomaly(earth.e, earth.p, earth_f1)
r1_pqw, v1_pqw = orb_2_pqw(r1, earth_f1, earth.e, earth.p, earth.mu)
r1_eci, v1_eci = perif_2_eci(r1_pqw, v1_pqw, earth.inc, earth.raan, earth.aop)


# SciPy Propgation
scipy_start = time.perf_counter()
earth_rs, earth_vs = propogate_orbit(
    r1_eci, v1_eci, earth.mu, tspan=earth.period(earth.a, earth.mu), dt=86400/2)
scipy_end = time.perf_counter()

earth_rs = earth_rs[0:-1]
earth_vs = earth_vs[0:-1]

# My RK4


def RK4_single_step(fun, dt, t0, y0, fun_arg: list):

    k1 = fun(t0, y0, fun_arg)
    k2 = fun((t0 + (dt/2)), (y0 + ((dt/2)*k1)), fun_arg)
    k3 = fun((t0 + (dt/2)), (y0 + ((dt/2)*k2)), fun_arg)
    k4 = fun((t0 + dt), (y0 + (dt*k3)), fun_arg)

    y1 = y0 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y1


def y_dot(t, y, fun_arg):
    """
    Two Body physics used for propagation
    """
    mu = fun_arg[0]
    rx, ry, rz, vx, vy, vz = y  # Deconstruct State to get r_vec
    print(t)
    r = np.array([rx, ry, rz])
    r_norm = np.linalg.norm(r)
    ax, ay, az = -r*mu/r_norm**3  # Two body Problem ODE
    return np.array([vx, vy, vz, ax, ay, az])


RK4_start = time.perf_counter()
t0 = 0
tf = earth.period(earth.a, earth.mu)
dt = 86400/2
n_steps = int((tf-t0)/dt)
ys = np.zeros((n_steps, 6))
ts = np.arange(t0, tf, dt)
y0 = np.concatenate((r1_eci, v1_eci))
fun_arg = [earth.mu]
ys[0] = y0
ts[0] = t0
step = 1

for i in range(np.size(ts)-2):
    ys[step] = RK4_single_step(
        y_dot, dt, ts[step-1], ys[step-1], fun_arg=fun_arg)
    step += 1
RK4_end = time.perf_counter()
earth_rs2 = ys[:, :3]
earth_vs2 = ys[:, 3:6]

"""
Plotting
"""
h1 = np.zeros(np.size(earth_rs[:, 0]))
for i in range(np.size(h1)-1):
    h1[i] = np.linalg.norm(np.cross(earth_rs[i], earth_vs[i]))

h2 = np.zeros(np.size(earth_rs2[:, 0]))
for i in range(np.size(h2)-1):
    h2[i] = np.linalg.norm(np.cross(earth_rs2[i], earth_vs2[i]))

print(f'Scipy took {scipy_end-scipy_start} seconds')
print(f'RK4 took {RK4_end-RK4_start} seconds')

h_intial = np.linalg.norm(np.cross(r1_eci, v1_eci))
h_f_scipy = np.linalg.norm(np.cross(earth_rs[-1], earth_vs[-1]))
h_f_rk4 = np.linalg.norm(np.cross(earth_rs2[-1], earth_vs2[-1]))

print(f'Intial h: {h_intial}')
print(f'Final h of scipy propagation: {h_f_scipy}')
print(f'Error scipy-intial: {h_f_scipy-h_intial}')
print(f'Final h of RK4 propagation: {h_f_rk4}')
print(f'Error RK4-intial: {h_f_rk4-h_intial}')

plt.figure()
plt.plot(h1[0:-1], color='orange', label='scipy')
plt.plot(h2[0:-1], color='blue', linestyle='--', label='rk4')
plt.xlabel("Time")
plt.ylabel("h")
plt.title("Compare Scipy To RK4")
plt.legend(loc='right')
plt.grid(True)
plt.show()


# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(earth_rs[:, 0], earth_rs[:, 1], earth_rs[:, 2], color='green')
# ax.plot(earth_rs2[:, 0], earth_rs2[:, 1], earth_rs2[:, 2], color='purple')
# plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.plot(earth_rs[330:360, 0], earth_rs[330:360, 1],
        earth_rs[330:360, 2], color='green')
ax.plot(earth_rs2[330:360, 0], earth_rs2[330:360, 1],
        earth_rs2[330:360, 2], color='purple', linestyle='--')
plt.show()

# Plot error?
ts = ts[0:-1]
ax1 = plt.subplot(3, 1, 1)
ax1.plot(earth_rs2[:, 0] - earth_rs[:, 0])

ax2 = plt.subplot(3, 1, 2)
ax2.plot(earth_rs2[:, 1] - earth_rs[:, 1])

ax1 = plt.subplot(3, 1, 3)
ax1.plot(earth_rs2[:, 2] - earth_rs[:, 2])

plt.show()

r1 = np.zeros(np.size(earth_rs[:, 0]))
for i in range(np.size(r1)-1):
    r1[i] = np.linalg.norm(earth_rs[i])

r2 = np.zeros(np.size(earth_rs2[:, 0]))
for i in range(np.size(r1)-1):
    r2[i] = np.linalg.norm(earth_rs2[i])

plt.figure()
plt.plot(r1[0:-1], color='orange', label='scipy')
plt.plot(r1[0:-1], color='blue', linestyle='--', label='rk4')
plt.xlabel("Time")
plt.ylabel("r_mag")
plt.title("Compare Scipy To RK4")
plt.legend(loc='right')
plt.grid(True)
plt.show()
