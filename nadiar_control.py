# Custom libs
from orbit import *
from Orbit_util import *
from body import *
from Universal_Variable import *
from quaternion import *

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

# Sets up single Runge Kutta 4 Step


def RK4_single_step(fun, dt, t0, y0, fun_arg: list):
    # evaluates inputted function, fun, at t0, y0, and inputted args to create 4 constants to solve 1 rk4 step
    # inputted function name --> y_dot_n_ephemeris
    k1 = fun(t0, y0, fun_arg)
    k2 = fun((t0 + (dt/2)), (y0 + ((dt/2)*k1)), fun_arg)
    k3 = fun((t0 + (dt/2)), (y0 + ((dt/2)*k2)), fun_arg)
    k4 = fun((t0 + dt), (y0 + (dt*k3)), fun_arg)

    y1 = y0 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y1

# sat_w_n+1 = sat_w_n + sat_w_dot*dt


def skew_mtx(x_vec):
    x1 = x_vec[0]
    x2 = x_vec[1]
    x3 = x_vec[2]

    return np.array([[0, -x3,  x2],
                     [x3,   0, -x1],
                     [-x2,  x1,   0]
                     ])


def euler_321(psi, theta, phi):
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

    return np.array([A11, A12, A13],
                    [A21, A22, A23],
                    [A31, A32, A33])

# phi_3, theta_2, psi_1


def euler321_to_quat(phi, theta, psi):
    # Appendix B: 3-2-1 -> quaternion
    q1 = np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2) - \
        np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)

    q2 = np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2) + \
        np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)

    q3 = np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2) - \
        np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2)

    q4 = np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2) + \
        np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)

    return Quaternion(np.array([q1, q2, q3, q4]).reshape(4, 1)).normalized()


def quat_to_attitude(q):

    q1, q2, q3, q4 = q.q1, q.q2, q.q3, q.q4

    A11 = (q1**2) - (q2**2) - (q3**2) + (q4**2)
    A12 = 2*(q1*q2 + q3*q4)
    A13 = 2*(q1*q3 - q2*q4)
    A21 = 2*(q2*q1 - q3*q4)
    A22 = -(q1**2) + (q2**2) - (q3**2) + (q4**2)
    A23 = 2*(q2*q3 + q1*q4)
    A31 = 2*(q3*q1 + q2*q4)
    A32 = 2*(q3*q2 - q1*q4)
    A33 = -(q1**2) - (q2**2) + (q3**2) + (q4**2)

    return np.array([
        [A11, A12, A13],
        [A21, A22, A23],
        [A31, A32, A33],
    ])

# https://en.wikiversity.org/wiki/PlanetPhysics/Direction_Cosine_Matrix_to_Euler_321_Angles
# phi_1, theta_2, psi_3


def attitude_to_euler321(attitude):
    A11 = attitude[0][0]
    A12 = attitude[0][1]
    A13 = attitude[0][2]
    A23 = attitude[1][2]
    A33 = attitude[2][2]

    theta = np.arccos(A13)
    phi = np.arctan(A12/A11)
    psi = np.arctan(A23, A33)

    return np.array([theta, phi, psi])

# Sarabandi, Soheil & Thomas, Federico. (2018). A Survey on the Computation of Quaternions From Rotation Matrices. Journal of Mechanisms and Robotics. 11. 10.1115/1 4041889.


def cayley_method(attitude):
    A11 = attitude[0][0]
    A12 = attitude[0][1]
    A13 = attitude[0][2]
    A21 = attitude[1][0]
    A22 = attitude[1][1]
    A23 = attitude[1][2]
    A31 = attitude[2][0]
    A32 = attitude[2][1]
    A33 = attitude[2][2]

    q4 = 0.25*np.sqrt((A11+A22+A33+1)**2 + (A32-A23) **
                      2 + (A13-A31)**2 + (A21-A12)**2)

    q1 = 0.25*np.sqrt((A32-A23)**2 + (A11-A22-A33+1) **
                      2 + (A21+A12)**2 + (A31+A13)**2)

    q2 = 0.25*np.sqrt((A13-A31)**2 + (A21+A12)**2 +
                      (A22-A11-A33+1)**2 + (A32+A23)**2)

    q3 = 0.25*np.sqrt((A21-A12)**2 + (A31+A13)**2 +
                      (A32+A23)**2 + (A33-A11-A22+1)**2)

    # if np.sign(q1) != np.sign((A32-A23)):
    #     q1 *= -1

    q1 = np.copysign(q1, (A32-A23))

    # if np.sign(q2) != np.sign((A13-A31)):
    #     q2 *= -1

    q2 = np.copysign(q2, (A13-A31))

    # if np.sign(q3) != np.sign((A21-A12)):
    #     q3 *= -1

    q3 = np.copysign(q3, (A21-A12))

    return Quaternion(np.array([q1, q2, q3, q4]).reshape(4, 1))


class ReactionWh:

    def __init__(self, mass, J_spin, J_perp, wl_unit):
        self.mass = mass
        self.J_spin = J_spin
        self.J_perp = J_perp
        self.wl_unit = wl_unit
        self.wl = 0

    # Inertia of non-spin axises of reachtion wheel
    def Jwh_body_perp(self):
        return self.J_perp*(np.identity(3) - np.outer(self.wl_unit, self.wl_unit))

    # Hwh_body = angular momentum of wheels in body frame
    def Hwh_body(self, sat_w):
        return self.J_spin*(self.wl_unit*sat_w + self.wl)*self.wl_unit

# EOM


def sat_dynamics(sat_h, sat_w, Lwh_b):
    return np.linalg.inv(JB)@(-Lwh_b-(np.cross(sat_w, sat_h, axis=0)))


# Exporting earth_orbit sim data
result = np.load("mission_data.npz", allow_pickle=True)
sat = result['arr_0'][()]
celestial_bodies = result['arr_1']
earth_orbit = np.load("leg_1_data.npy", allow_pickle=True)[()]
central_body = earth_orbit["central_body"]
sat_orbit = earth_orbit["sat_orbit"]
bodies = earth_orbit["bodies"]
orbit_dt = earth_orbit["dt"]
orbit_n_steps_1 = earth_orbit["n_steps"]
orbit_t0 = earth_orbit["t0"]
orbit_tf = earth_orbit["tf"]
orbit_y0 = earth_orbit["y0"]
orbit_ts = earth_orbit["ts"]
orbit_ys = earth_orbit["ys"]

sat.inertia = np.array([[40, 0, 0],
                        [0, 50, 0],
                        [0, 0, 60]
                        ])

r = orbit_ys[0, 0:3]
v = orbit_ys[0, 3:6]

# Define LVLH frame
O1 = -r/np.linalg.norm(r)
O2 = -np.cross(r, v)/np.linalg.norm(np.cross(r, v))
O3 = np.cross(O1, O2)

lvlh_attitude = np.stack([O1, O2, O3], axis=1)


# Define Reaction wheels
wh1 = ReactionWh(1, 3, 6, np.array([1, 0, 0]).reshape(3, 1))
wh2 = ReactionWh(1, 3, 6, np.array([0, 1, 0]).reshape(3, 1))
wh3 = ReactionWh(1, 3, 6, np.array([0, 0, 1]).reshape(3, 1))

# JB = inertia of S/C with Jwh_body
JB = sat.inertia + wh1.Jwh_body_perp() + wh2.Jwh_body_perp() + \
    wh3.Jwh_body_perp()

# Sim var
kp = 10
kd = 20

t0 = 0
tf = 50
dt = .25
ts = np.arange(t0, tf + dt, dt)
n_steps = len(ts)
ys = np.zeros((n_steps, 7))

q_sat_hist = np.zeros((n_steps, 4))
q_error_hist = np.zeros((n_steps, 4))
sat_w_hist = np.zeros((n_steps, 3))
L_hist = np.zeros((n_steps, 3))

# Commanded sat position
phi_c = 10
theat_c = 1
psi_c = 6
q_c = euler321_to_quat(np.deg2rad(
    phi_c), np.deg2rad(theat_c), np.deg2rad(psi_c))

# intial sat position
phi0 = 60
theat0 = 45
psi0 = 30
q_sat0 = euler321_to_quat(np.deg2rad(
    phi0), np.deg2rad(theat0), np.deg2rad(psi0))

# sat_w = angular momentum of body relative to inertial in body frame
sat_w0 = np.array([3, -7, 2]).reshape(3, 1)

R = quat_to_attitude(q_sat0)
Rq = cayley_method(R)
RqA = quat_to_attitude(Rq)

y0 = np.concatenate([q_sat0.value.flatten(), sat_w0.flatten()])

ys[0] = y0
ts[0] = t0


def y_dot_gnc(t, y, fun_arg: list):
    q_sat = Quaternion(y[0:4].reshape(4, 1))
    q_sat = q_sat.normalized()
    sat_w = y[4:].reshape(3, 1)
    JB = fun_arg[0]
    Lwh_b = fun_arg[1]

    total_wh_h = wh1.Hwh_body(
        sat_w) + wh2.Hwh_body(sat_w) + wh3.Hwh_body(sat_w)
    sat_h = (JB@sat_w) + total_wh_h
    sat_w_dot = sat_dynamics(sat_h, sat_w, Lwh_b)

    q_sat_dot = q_sat.kinamatics(sat_w)

    return np.concatenate([q_sat_dot.value.flatten(), sat_w_dot.flatten()])


for i in range(n_steps-1):
    # For calcs
    q_sat = Quaternion(ys[i][0:4].reshape(4, 1))
    q_sat = q_sat.normalized()
    sat_w = ys[i][4:].reshape(3, 1)

    # Calc Control Moment
    q_error = q_sat.cross(q_c.inverse())
    Lwh_b = (-kp*q_error.vector - kd*sat_w)*-1

    # Propgate one step w/ Control Moment
    ys[i+1] = RK4_single_step(y_dot_gnc, dt, ts[i], ys[i], fun_arg=[JB, Lwh_b])

    # For plotting
    q_sat_hist[i] = ys[i][0:4].flatten()
    sat_w_hist[i] = ys[i][4:].flatten()
    q_error_hist[i] = q_error.value.flatten()
    L_hist[i] = Lwh_b.flatten()

    # For plotting
    q_sat_hist[i+1] = ys[i][0:4].flatten()
    sat_w_hist[i+1] = ys[i][4:].flatten()
    q_error_hist[i+1] = q_error.value.flatten()
    L_hist[i+1] = Lwh_b.flatten()
