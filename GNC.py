# Custom libs
from orbit import *
from Orbit_util import *
from body import *
from Universal_Variable import *

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
    k2 = fun((t0 + (dt/2)), (y0 + ((dt.value/2)*k1)), fun_arg)
    k3 = fun((t0 + (dt/2)), (y0 + ((dt.value/2)*k2)), fun_arg)
    k4 = fun((t0 + dt), (y0 + (dt.value*k3)), fun_arg)
    y1 = y0 + (dt.value/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y1


class Quaternion:

    # q = [vector, scalor]
    def __init__(self, q1, q2, q3, q4):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.scalor = q4
        self.vector = np.array([q1, q2, q3]).reshape(3, 1)
        self.value = np.array(
            [self.q1, self.q2, self.q3, self.q4]).reshape(4, 1)

    def inverse(self):
        q_mag = np.linalg.norm(self.value)
        q_conj = np.array([-self.q1, -self.q2, -self.q3, self.q4])
        q_inv = q_conj / (q_mag**2)

        return Quaternion(q_inv[0], q_inv[1], q_inv[2], q_inv[3])

    @staticmethod
    def identity():
        return Quaternion(np.zeros(3), 1)

    @staticmethod
    def phi(q):
        if not isinstance(q, Quaternion):
            raise TypeError("Expected Quaternion")

        q1 = q.value[0][0]
        q2 = q.value[1][0]
        q3 = q.value[2][0]
        q4 = q.value[3][0]

        return np.array([
            [q4,  q3, -q2],
            [-q3,  q4,  q1],
            [q2, -q1,  q4],
            [-q1, -q2, -q3]
        ])

    @staticmethod
    def eps(q):
        if not isinstance(q, Quaternion):
            raise TypeError("Expected Quaternion")

        q1 = q.value[0][0]
        q2 = q.value[1][0]
        q3 = q.value[2][0]
        q4 = q.value[3][0]

        return np.array([
            [q4,  -q3, q2],
            [q3,  q4,  -q1],
            [-q2, q1,  q4],
            [-q1, -q2, -q3]
        ])

    def cross(self, q2):
        result = np.block([Quaternion.phi(self), self.value]) @ q2.value
        return Quaternion(result[0], result[1], result[2], result[3])

    def dot(self, q2):
        result = np.block([Quaternion.eps(self), self.value]) @ q2.value
        return Quaternion(result[0], result[1], result[2], result[3])


def skew_mtx(x_vec):
    x1 = x_vec[0]
    x2 = x_vec[1]
    x3 = x_vec[2]

    return np.array([[0, -x3,  x2],
                     [x3,   0, -x1],
                     [-x2,  x1,   0]
                     ])


def euler_313(psi, theta, phi):
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


def euler_to_quat(psi, theta, phi):
    # Appendix B: 3-2-1 -> quaternion
    q1 = np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2) - \
        np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)

    q2 = np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2) + \
        np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)

    q3 = np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2) - \
        np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2)

    q4 = np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2) + \
        np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)

    return Quaternion(q1, q2, q3, q4)


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
dt = earth_orbit["dt"]
n_steps_1 = earth_orbit["n_steps"]
t0 = earth_orbit["t0"]
tf = earth_orbit["tf"]
y0 = earth_orbit["y0"]
ts = earth_orbit["ts"]
ys = earth_orbit["ys"]

# Define sat and principle axis
sat.inertia = np.array([[40, 0, 0],
                        [0, 50, 0],
                        [0, 0, 60]
                        ])

D, V = np.linalg.eig(sat.inertia)

S1 = V[:, 0]
S2 = V[:, 1]
S3 = V[:, 2]

r = ys[0, 0:3]
v = ys[0, 3:6]

# Define LVLH frame
O1 = -r/np.linalg.norm(r)
O2 = -np.cross(r, v)/np.linalg.norm(np.cross(r, v))
O3 = np.cross(O1, O2)

# Define Reaction wheels
wh1 = ReactionWh(1, 3, 6, np.array([1, 0, 0]).reshape(3, 1))
wh2 = ReactionWh(1, 3, 6, np.array([0, 1, 0]).reshape(3, 1))
wh3 = ReactionWh(1, 3, 6, np.array([0, 0, 1]).reshape(3, 1))

# JB = inertia of S/C with Jwh_body
JB = sat.inertia + wh1.Jwh_body_perp() + wh2.Jwh_body_perp() + \
    wh3.Jwh_body_perp()

# Sim vars
t0 = 0
tf = 200
dt = 0.01
ts = np.arange(t0, tf + dt, dt)

# Commanded sat position
psi_c = 30
theat_c = 0
phi_c = 0
q_c = euler_to_quat(np.deg2rad(
    psi_c), np.deg2rad(theat_c), np.deg2rad(phi_c))

# intial sat position
psi0 = 60
theat0 = 0
phi0 = 0
q_sat0 = euler_to_quat(np.deg2rad(psi0), np.deg2rad(theat0), np.deg2rad(phi0))
# sat_w = angular momentum of body relative to inertial in body frame
sat_w0 = np.array([3, 0, 0]).reshape(3, 1)

q_error_0 = q_sat0.cross(q_c.inverse())

kp = 50
kd = 100

q_sat_hist = np.zeros((len(ts), 4))
L_hist = np.zeros((len(ts), 3))
q_error = q_error_0
q_sat = q_sat0
sat_w = sat_w0

for i, t in enumerate(ts):
    q_sat_hist[i] = q_sat.value.reshape(4)

    Lwh_b = kp*q_error.vector + kd*sat_w
    L_hist[i] = Lwh_b.reshape(3)

    # Total Angular momentum of sat with wheels
    total_wh_h0 = wh1.Hwh_body(
        sat_w) + wh2.Hwh_body(sat_w) + wh3.Hwh_body(sat_w)

    # total_wh_h0 = (wh1.Hwh_body(sat_w) + (Lwh_b*wh1.wl_unit)*dt) + \
    # (wh2.Hwh_body(sat_w) + (Lwh_b*wh2.wl_unit)*dt) + \
    # (wh3.Hwh_body(sat_w) + (Lwh_b*wh3.wl_unit)*dt)

    sat_h = (JB@sat_w) + total_wh_h0

    # EOM of sat
    sat_w_dot = sat_dynamics(sat_h, sat_w, Lwh_b)

    sat_w = sat_w + sat_w_dot*dt

    # Dynamics of Quaternion
    q_sat_dot = 0.5*Quaternion.eps(q_sat) @ sat_w

    q_sat = q_sat.value + q_sat_dot*dt
    q_sat = Quaternion(q_sat[0], q_sat[1], q_sat[2], q_sat[3])

    q_error = q_sat.cross(q_c.inverse())


print("done")

# # Create a 3D plot
# ax = plt.figure().add_subplot(projection='3d')
# origin = np.array([0, 0, 0])
# O_colors = ['#D62828', '#003049', '#F77F00']
# S_colors = ['#FF6B9D', '#00B4D8', '#FFD60A']

# # LVLH FRAME
# ax.quiver(origin[0], origin[1], origin[2],
#           O1[0], O1[1], O1[2],
#           color=O_colors[0], arrow_length_ratio=0.15, linewidth=2, label='O1')

# ax.quiver(origin[0], origin[1], origin[2],
#           O2[0], O2[1], O2[2],
#           color=O_colors[1], arrow_length_ratio=0.15, linewidth=2, label='O2')

# ax.quiver(origin[0], origin[1], origin[2],
#           O3[0], O3[1], O3[2],
#           color=O_colors[2], arrow_length_ratio=0.15, linewidth=2, label='O3')

# # SPACECRAFT PRINCIPAL AXIS FRAME
# ax.quiver(origin[0], origin[1], origin[2],
#           S1[0], S1[1], S1[2],
#           color=S_colors[0], arrow_length_ratio=0.15, linewidth=2, label='S1', linestyle='--')

# ax.quiver(origin[0], origin[1], origin[2],
#           S2[0], S2[1], S2[2],
#           color=S_colors[1], arrow_length_ratio=0.15, linewidth=2, label='S2', linestyle='--')

# ax.quiver(origin[0], origin[1], origin[2],
#           S3[0], S3[1], S3[2],
#           color=S_colors[2], arrow_length_ratio=0.15, linewidth=2, label='S3', linestyle='--')

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Orbital Reference Frame (O1, O2, O3)')
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.legend()
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()
