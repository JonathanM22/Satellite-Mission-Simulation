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


class Quaternion:

    # q = [vector, scalor]
    def __init__(self, q1, q2, q3, q4):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.scalor = q4
        self.vector = np.array([q1, q2, q3])
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

        q1, q2, q3, q4 = q.q1, q.q2, q.q3, q.q4

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

        q1, q2, q3, q4 = q.q1, q.q2, q.q3, q.q4

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
        return self.J_spin*(np.dot(self.wl_unit, sat_w) + self.wl)*self.wl_unit


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
wh1 = ReactionWh(1, 3, 6, np.array([1, 0, 0]))
wh2 = ReactionWh(1, 3, 6, np.array([0, 1, 0]))
wh3 = ReactionWh(1, 3, 6, np.array([0, 0, 1]))

# JB = inertia of S/C with Jwh_body
JB = sat.inertia + wh1.Jwh_body_perp() + wh2.Jwh_body_perp() + \
    wh3.Jwh_body_perp()


# intial sat position
psi0 = 60
theat0 = 0
phi0 = 0
q_sat = euler_to_quat(np.deg2rad(psi0), np.deg2rad(theat0), np.deg2rad(phi0))
sat_w = np.array([3, 0, 0])


# Commanded sat position
psi_c = 30
theat_c = 0
phi_c = 0
q_c = euler_to_quat(np.deg2rad(
    psi_c), np.deg2rad(theat_c), np.deg2rad(phi_c))

q_error = q_sat.cross(q_c.inverse())
print(q_error.value)

sat_w = np.array([3, 0, 0])

# Total Angular momentum of sat with wheels
total_wh_h = wh1.Hwh_body(sat_w) + wh2.Hwh_body(sat_w) + wh3.Hwh_body(sat_w)
sat_h = (JB*sat_w) + total_wh_h

# EOM
# sat_w = angular momentum of body relative to inertial in body frame


def sat_w_dot(JB, Lwh_b, sat_w, total_wh_h):
    return np.linalg.inv(JB)*(-Lwh_b - (np.cross(sat_w, (JB*sat_w + total_wh_h))))


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
