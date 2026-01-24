# Custom libs
from orbit import *
from Orbit_util import *
from body import *
from Universal_Variable import *
from quaternion import *
from reactionwh import *

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


def euler_321_attitude(phi, theta, psi):
    # Appendix B 3-2-1 matrix OR sec 2.9 pg55
    A11 = np.cos(theta)*np.cos(phi)
    A12 = np.cos(theta)*np.sin(phi)
    A13 = -np.sin(theta)
    A21 = (-np.cos(psi)*np.sin(phi)) + (np.sin(psi)*np.sin(theta)*np.cos(phi))
    A22 = (np.cos(psi)*np.cos(phi)) + (np.sin(psi)*np.sin(theta)*np.sin(phi))
    A23 = np.sin(psi)*np.cos(theta)
    A31 = (np.sin(psi)*np.sin(phi)) + (np.cos(psi)*np.sin(theta)*np.cos(phi))
    A32 = (-np.sin(psi)*np.cos(phi)) + (np.cos(psi)*np.sin(theta)*np.sin(phi))
    A33 = np.cos(psi)*np.cos(theta)

    return np.array([[A11, A12, A13],
                     [A21, A22, A23],
                     [A31, A32, A33]])


def attitude_to_euler321(attitude):
    # https://en.wikiversity.org/wiki/PlanetPhysics/Direction_Cosine_Matrix_to_Euler_321_Angles
    # phi_1, theta_2, psi_3
    A11 = attitude[0][0]
    A12 = attitude[0][1]
    A13 = attitude[0][2]
    A23 = attitude[1][2]
    A33 = attitude[2][2]

    theta = np.arcsin(-A13)
    phi = np.atan2(A12, A11)
    psi = np.atan2(A23, A33)

    return np.array([phi, theta, psi])


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


def skew_mtx(x_vec):
    x1 = x_vec[0]
    x2 = x_vec[1]
    x3 = x_vec[2]

    return np.array([[0, -x3,  x2],
                     [x3,   0, -x1],
                     [-x2,  x1,   0]
                     ])


def unpack_state_ar(yi):
    # Unpacks a state to components for calculations
    r = yi[0:3]
    v = yi[3:6]
    q_sat = Quaternion(yi[6:10].reshape(4, 1)).normalized()
    sat_w = yi[10:].reshape(3, 1)

    return r, v, q_sat, sat_w


def rv_to_lvlh(r, v):
    # Define LVLH frame
    O1 = -r/np.linalg.norm(r)
    O2 = -np.cross(r, v)/np.linalg.norm(np.cross(r, v))
    O3 = np.cross(O1, O2)

    return np.stack([O1, O2, O3], axis=1)


def sat_dynamics(sat_h, sat_w, Lwh_b):
    # EOM
    return np.linalg.inv(JB)@(-Lwh_b-(np.cross(sat_w, sat_h, axis=0)))


# Define Earth Parking Orbit
earth_parking = Orbit(mu=EARTH_MU,
                      a=32000*u.km,
                      e=0.80*u.km/u.km,  # unitless
                      f0=(180*u.deg).to(u.rad),
                      inc=(28*u.deg).to(u.rad),
                      raan=(175*u.deg).to(u.rad),
                      aop=(240*u.deg).to(u.rad)
                      )

earth_parking.p = earth_parking.calc_p(earth_parking.a, earth_parking.e)
earth_parking.energy = earth_parking.calc_energy(
    earth_parking.a, earth_parking.mu)

# Intialize SAT
SAT_MASS = 100*u.kg
sat = Spacecraft(SAT_MASS, epoch, label="sat", color="purple")
sat.inertia = np.array([[40, 0, 0],
                        [0, 50, 0],
                        [0, 0, 60]
                        ])
sat_orbit = earth_parking

# Getting intial position & vel. This r is wrt to the central body: earth
r = sat_orbit.r_at_true_anomaly(sat_orbit.e, sat_orbit.p, sat_orbit.f0)

# Everything is in km, for numpy to work you need to have float numbers
r_pqw, v_pqw = orb_2_pqw(r.value,
                         sat_orbit.f0.value, sat_orbit.e.value,
                         sat_orbit.p.value, sat_orbit.mu.value)

r_eci, v_eci = perif_2_eci(r_pqw, v_pqw, sat_orbit.inc.value,
                           sat_orbit.raan.value, sat_orbit.aop.value)

# both of these are relative to the CB: earth
sat.r0 = r_eci * u.km
sat.v0 = v_eci * (u.km/u.s)

# Define Reaction wheels
wh1 = ReactionWh(1, 3, 6, np.array([1, 0, 0]).reshape(3, 1))
wh2 = ReactionWh(1, 3, 6, np.array([0, 1, 0]).reshape(3, 1))
wh3 = ReactionWh(1, 3, 6, np.array([0, 0, 1]).reshape(3, 1))

# JB = inertia of S/C with Jwh_body
JB = sat.inertia + wh1.Jwh_body_perp() + wh2.Jwh_body_perp() + \
    wh3.Jwh_body_perp()

# Control Law constants
kp = 4
kd = 8

# Sim central and kth bodies
central_body = earth
bodies = [moon]

# Setting Up Time series
epoch = Time("2026-11-08")
dt = TimeDelta(0.1, format='sec')
dt_steps = np.round(
    (sat_orbit.period(sat_orbit.a, sat_orbit.mu).value/20)/dt.value)
t0 = epoch - dt*dt_steps + dt
tf = epoch
ts = np.arange(t0, tf+dt, dt)
n_steps = len(ts)
ys = np.zeros((n_steps, 13))

# Intial sat position
phi0 = 60
theat0 = 45
psi0 = 30
q_sat0 = Quaternion.from_euler321(np.deg2rad(
    phi0), np.deg2rad(theat0), np.deg2rad(psi0))

# Initial sat_w = angular momentum of body relative to inertial in body frame
sat_w0 = np.array([3, -7, 2]).reshape(3, 1)

y0 = np.concatenate([sat.r0.value, sat.v0.value,
                    q_sat0.value.flatten(), sat_w0.flatten()])

ys[0] = y0
ts[0] = t0


def y_dot_nadir(t, y, fun_arg: list):

    # Unpack state
    r, v, q_sat, sat_w = unpack_state_ar(y)

    # Unpack fun_arg
    central_body = fun_arg[0]
    bodies = fun_arg[1]
    JB = fun_arg[2]
    Lwh_b = fun_arg[3]

    # Calc r_dot, v_dot
    r_mag = np.linalg.norm(r)
    r_c = get_body_barycentric(central_body.label, t).xyz.to(u.km).value
    m_c = central_body.mass.value
    a = ((central_body.mu.value)/(r_mag**3)) * -r

    for body in bodies:
        r_k = get_body_barycentric(body.label, t).xyz.to(u.km).value
        m_k = body.mass.value
        r_ck = r_k - r_c
        r_sk = r_ck - r
        r_sk_mag = np.linalg.norm(r_sk)
        r_ck_mag = np.linalg.norm(r_ck)
        a_k = ((body.mu.value)/(r_sk_mag**3)) * r_sk
        a_cb_k = ((body.mu.value)/(r_ck_mag**3)) * r_ck
        a += a_k - a_cb_k

    # Calc sat_w_dot
    total_wh_h = wh1.Hwh_body(
        sat_w) + wh2.Hwh_body(sat_w) + wh3.Hwh_body(sat_w)
    sat_h = (JB@sat_w) + total_wh_h
    sat_w_dot = sat_dynamics(sat_h, sat_w, Lwh_b)

    # Calc q_sat_dot
    q_sat_dot = q_sat.kinamatics(sat_w)

    return np.concatenate([v, a, q_sat_dot.value.flatten(), sat_w_dot.flatten()])


# For Plotting
q_error_hist = np.zeros((n_steps, 4))
L_hist = np.zeros((n_steps, 3))
lvlh_attitude_hist = np.zeros((n_steps, 3, 3))

# Propgation
start_time = time.perf_counter()
for i in range(n_steps-1):

    # Unpack state
    r, v, q_sat, sat_w = unpack_state_ar(ys[i])

    # Calc Control Moment
    lvlh_attitude = rv_to_lvlh(r, v)
    q_c = Quaternion.from_attitude(lvlh_attitude)
    q_error = q_sat.cross(q_c.inverse())
    Lwh_b = (-kp*q_error.vector - kd*sat_w)*-1

    # Propgate one step w/ Control Moment
    ys[i+1] = RK4_single_step(y_dot_nadir, dt, ts[i],
                              ys[i], fun_arg=[central_body, bodies, JB, Lwh_b])

    # For plotting
    q_error_hist[i] = q_error.value.flatten()
    L_hist[i] = Lwh_b.flatten()
    lvlh_attitude_hist[i, :, :] = lvlh_attitude

end_time = time.perf_counter()

# For plotting
q_error_hist[i+1] = q_error.value.flatten()
L_hist[i+1] = Lwh_b.flatten()
lvlh_attitude_hist[i+1, :, :] = lvlh_attitude

r_hist = np.zeros((n_steps, 3))
v_hist = np.zeros((n_steps, 3))
q_sat_hist = np.zeros((n_steps, 4))
sat_w_hist = np.zeros((n_steps, 3))

for i, state in enumerate(ys):
    r, v, q_sat, sat_w = unpack_state_ar(state)
    r_hist[i] = r
    v_hist[i] = v
    q_sat_hist[i] = q_sat.value.flatten()
    sat_w_hist[i] = sat_w.flatten()

data_dict = {
    "central_body": central_body,
    "sat_orbit": sat_orbit,
    "bodies": bodies,
    "sat": sat,
    "reaction_wheels": [wh1, wh2, wh3],
    "epoch": epoch,
    "dt": dt,
    "n_steps": n_steps,
    "t0": t0,
    "tf": tf,
    "y0": y0,
    "ts": ts,
    "ys": ys,
    "q_error_hist": q_error_hist,
    "L_hist": L_hist,
    "r_hist": r_hist,
    "v_hist": v_hist,
    "q_sat_hist": q_sat_hist,
    "sat_w_hist": sat_w_hist,
    "lvlh_attitude_hist": lvlh_attitude_hist,
    "kp": kp,
    "kd": kd,
    "JB": JB
}

nadir_control_sim_data = np.save("nadir_control_sim_data", np.array(data_dict))

print(q_error_hist[-10:-1])
