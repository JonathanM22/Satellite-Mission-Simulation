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


O1 = -r/np.linalg.norm(r)
O2 = -np.cross(r, v)/np.linalg.norm(np.cross(r, v))
O3 = np.cross(O1, O2)


# Create a 3D plot
ax = plt.figure().add_subplot(projection='3d')
origin = np.array([0, 0, 0])
O_colors = ['#D62828', '#003049', '#F77F00']
S_colors = ['#FF6B9D', '#00B4D8', '#FFD60A']

# LVLH FRAME
ax.quiver(origin[0], origin[1], origin[2],
          O1[0], O1[1], O1[2],
          color=O_colors[0], arrow_length_ratio=0.15, linewidth=2, label='O1')

ax.quiver(origin[0], origin[1], origin[2],
          O2[0], O2[1], O2[2],
          color=O_colors[1], arrow_length_ratio=0.15, linewidth=2, label='O2')

ax.quiver(origin[0], origin[1], origin[2],
          O3[0], O3[1], O3[2],
          color=O_colors[2], arrow_length_ratio=0.15, linewidth=2, label='O3')

# SPACECRAFT PRINCIPAL AXIS FRAME
ax.quiver(origin[0], origin[1], origin[2],
          S1[0], S1[1], S1[2],
          color=S_colors[0], arrow_length_ratio=0.15, linewidth=2, label='S1', linestyle='--')

ax.quiver(origin[0], origin[1], origin[2],
          S2[0], S2[1], S2[2],
          color=S_colors[1], arrow_length_ratio=0.15, linewidth=2, label='S2', linestyle='--')

ax.quiver(origin[0], origin[1], origin[2],
          S3[0], S3[1], S3[2],
          color=S_colors[2], arrow_length_ratio=0.15, linewidth=2, label='S3', linestyle='--')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Orbital Reference Frame (O1, O2, O3)')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
