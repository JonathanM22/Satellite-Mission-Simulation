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
              inc=np.deg2rad(175),
              raan=np.deg2rad(237),
              aop=np.deg2rad(102.9),
              mu=SUN_MU)
earth.p = earth.calc_p(earth.a, earth.e)

transfer = Orbit(mu=SUN_MU)

# Set Up r1 and r2 & TOF
earth_f1 = np.deg2rad(90)

r1 = earth.r_at_true_anomaly(earth.e, earth.p, earth_f1)
r1_pqw, v1_pqw = orb_2_pqw(r1, earth_f1, earth.e, earth.p, earth.mu)
r1_eci, v1_eci = perif_2_eci(r1_pqw, v1_pqw, earth.inc, earth.raan, earth.aop)

"""
a, e, e_vec, inc, raan, aop, f = rv_2_orb_elm(r1_eci, v1_eci, earth.mu)
print("Testing rv_2_orb_elm")
print(f"SMA: {earth.a-a}")
print(f"e: {earth.e-e}")
print(f"INC: {earth.inc-inc}")
print(f"RAAN: {earth.raan-raan}")
print(f"AOP: {earth.aop-aop}")
print(f"f: {earth_f1-f}")
print("---------------------------------")
"""


"""
SciPy Propgation
"""
scipy_start = time.perf_counter()
earth_rs, earth_vs = propogate_orbit(
    r1_eci, v1_eci, earth.mu, tspan=earth.period(earth.a, earth.mu), dt=86400/2)
scipy_end = time.perf_counter()

earth_rs = earth_rs[0:-1]
earth_vs = earth_vs[0:-1]

"""
My RK4 Propgation
"""


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
    # print(t)
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
Results
"""
print(f'------------------------------------------------------------------------')
print(f'Scipy took {scipy_end-scipy_start} seconds')
print(f'RK4 took {RK4_end-RK4_start} seconds')
print(f'------------------------------------------------------------------------')

h_intial = np.linalg.norm(np.cross(r1_eci, v1_eci))
h_f_scipy = np.linalg.norm(np.cross(earth_rs[-1], earth_vs[-1]))
h_f_rk4 = np.linalg.norm(np.cross(earth_rs2[-1], earth_vs2[-1]))

print(f'------------------------------------------------------------------------')
print(f'Intial h: {h_intial}')
print(f'Final h of scipy propagation: {h_f_scipy}')
print(f'Error scipy-intial: {h_f_scipy-h_intial}')
print(f'Final h of RK4 propagation: {h_f_rk4}')
print(f'Error RK4-intial: {h_f_rk4-h_intial}')
print(f'------------------------------------------------------------------------')

# Keeping track of all important elements for scipy
h1 = np.zeros(np.size(earth_rs[:, 0]))
a1 = np.zeros(np.shape(h1))
e1 = np.zeros(np.shape(h1))
inc1 = np.zeros(np.shape(h1))
raan1 = np.zeros(np.shape(h1))
aop1 = np.zeros(np.shape(h1))

for i in range(np.size(h1)-1):

    a, e, e_vec, inc, raan, aop, f = rv_2_orb_elm(
        earth_rs[i], earth_vs[i], earth.mu)

    h1[i] = np.linalg.norm(np.cross(earth_rs[i], earth_vs[i]))
    a1[i] = a
    e1[i] = e
    inc1[i] = inc
    raan1[i] = raan
    aop1[i] = aop


# Keeping track of all important elements for RK4
h2 = np.zeros(np.size(earth_rs2[:, 0]))
a2 = np.zeros(np.shape(h2))
e2 = np.zeros(np.shape(h2))
inc2 = np.zeros(np.shape(h2))
raan2 = np.zeros(np.shape(h2))
aop2 = np.zeros(np.shape(h2))

for i in range(np.size(h2)-1):
    a, e, e_vec, inc, raan, aop, f = rv_2_orb_elm(
        earth_rs2[i], earth_vs2[i], earth.mu)

    h2[i] = np.linalg.norm(np.cross(earth_rs2[i], earth_vs2[i]))
    a2[i] = a
    e2[i] = e
    inc2[i] = inc
    raan2[i] = raan
    aop2[i] = aop


ax1 = plt.subplot(3, 2, 1)
ax1.plot(h1[0:-1], color='orange', label='scipy')
ax1.plot(h2[0:-1], color='blue', linestyle='--', label='rk4')
ax1.set_xlabel("time (seconds)")
ax1.set_ylabel("h")
ax1.set_title("Angular Momentum Comparison")
ax1.legend(loc='lower left')
ax1.grid(True)

ax2 = plt.subplot(3, 2, 2)
ax2.plot(a1[0:-1], color='orange', label='scipy')
ax2.plot(a2[0:-1], color='blue', linestyle='--', label='rk4')
ax2.set_xlabel("time (seconds)")
ax2.set_ylabel("a")
ax2.set_title("SMA Comparison")
ax2.legend(loc='lower left')
ax2.grid(True)

ax3 = plt.subplot(3, 2, 3)
ax3.plot(e1[0:-1], color='orange', label='scipy')
ax3.plot(e2[0:-1], color='blue', linestyle='--', label='rk4')
ax3.set_xlabel("time (seconds)")
ax3.set_ylabel("e")
ax3.set_title("ECC Comparison")
ax3.legend(loc='lower left')
ax3.grid(True)

ax4 = plt.subplot(3, 2, 4)
ax4.plot(inc1[0:-1], color='orange', label='scipy')
ax4.plot(inc2[0:-1], color='blue', linestyle='--', label='rk4')
ax4.set_xlabel("time (seconds)")
ax4.set_ylabel("inc")
ax4.set_title("INC Comparison")
ax4.legend(loc='lower left')
ax4.grid(True)

ax5 = plt.subplot(3, 2, 5)
ax5.plot(raan1[0:-1], color='orange', label='scipy')
ax5.plot(raan2[0:-1], color='blue', linestyle='--', label='rk4')
ax5.set_xlabel("time (seconds)")
ax5.set_ylabel("raan")
ax5.set_title("RAAN Comparison")
ax5.legend(loc='lower left')
ax5.grid(True)

ax6 = plt.subplot(3, 2, 6)
ax6.plot(aop1[0:-1], color='orange', label='scipy')
ax6.plot(aop2[0:-1], color='blue', linestyle='--', label='rk4')
ax6.set_xlabel("time (seconds)")
ax6.set_ylabel("aop")
ax6.set_title("AOP Comparison")
ax6.legend(loc='lower left')
ax6.grid(True)

plt.legend()
plt.tight_layout()
plt.show()

print('break')
