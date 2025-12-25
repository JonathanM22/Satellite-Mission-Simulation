"""
ANIMATING PLOTS
"""

# Custom libs
from orbit import *
from Orbit_util import *
from body import *
from Universal_Variable import *

# Standard libs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from astropy.time import Time
from astropy.time import TimeDelta
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
from astropy.coordinates import get_body_barycentric

# Plot Setting
plot_leg_1 = True

# Unpacking Data
result = np.load("mission_data.npz", allow_pickle=True)
sat = result['arr_0'][()]
# celestial_bodies = [sun, earth, moon, mars, mercury, jupiter, venus, saturn, uranus, neptune]
celestial_bodies = result['arr_1']

"""
Plot LEG-1 Earth Depature
"""
leg_1_data = np.load("leg_1_data.npy", allow_pickle=True)[()]
n_steps_1 = leg_1_data["n_steps"]

start_index = 0
end_index = n_steps_1 + 10

earth = celestial_bodies[1]
moon = celestial_bodies[2]
earth_orbit = leg_1_data["sat_orbit"]


if plot_leg_1:

    moon_wrt_earth = moon.r_ar[start_index:end_index,
                               :] - earth.r_ar[start_index:end_index, :]
    sat_wrt_earth = sat.r_ar[start_index:end_index,
                             :] - earth.r_ar[start_index:end_index, :]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    n_frames = len(sat_wrt_earth)

    # Earth
    earth_point = ax.scatter(0, 0, 0, color=earth.color, s=5,
                             marker='o', edgecolor='k', label=earth.label)

    # Moon
    moon_line, = ax.plot([], [], [], color=moon.color, label=moon.label)
    moon_point, = ax.plot([], [], [], 'o', color=moon.color, markersize=3)

    # Sat
    sat_line, = ax.plot([], [], [], color=sat.color, label=sat.label)
    sat_point, = ax.plot([], [], [],  color=sat.color,
                         markersize=2,  markeredgecolor='black')

    ax.set_title(f"LEG-1: Earth Depature", fontsize=14, pad=10)
    ax.set_aspect('equal')
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_xlim([-300000, 300000])
    ax.set_ylim([-100000, 200000])
    ax.set_zlim([-300000, 300000])
    ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    ax.view_init(elev=32, azim=69, roll=0)
    plt.tight_layout()

    def update(frame):
        sat_line.set_data(sat_wrt_earth[:frame, 0], sat_wrt_earth[:frame, 1])
        sat_line.set_3d_properties(sat_wrt_earth[:frame, 2])
        sat_point.set_data([sat_wrt_earth[frame, 0]],
                           [sat_wrt_earth[frame, 1]])
        sat_point.set_3d_properties([sat_wrt_earth[frame, 2]])

        moon_line.set_data(
            moon_wrt_earth[:frame, 0], moon_wrt_earth[:frame, 1])
        moon_line.set_3d_properties(moon_wrt_earth[:frame, 2])
        moon_point.set_data([moon_wrt_earth[frame, 0]],
                            [moon_wrt_earth[frame, 1]])
        moon_point.set_3d_properties([moon_wrt_earth[frame, 2]])

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=10,    # ms per frame
        blit=False      # blitting is unreliable in 3D
    )

    # plt.show()
    ani.save("leg1_animation.gif", writer="pillow", fps=30)
    # ani.save("leg1_animation.gif", writer="pillow", fps=30)
