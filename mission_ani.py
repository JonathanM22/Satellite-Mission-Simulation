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
from matplotlib.animation import FFMpegWriter, PillowWriter
import time
from astropy.time import Time
from astropy.time import TimeDelta
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
from astropy.coordinates import get_body_barycentric

# Animation driver
# Change to reflect your file location!
plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\jonam\Desktop\Aero-Project\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

# Plot Setting
plot_leg_1 = False
plot_leg_2 = False
plot_leg_3 = True

# Unpacking Data
result = np.load("mission_data.npz", allow_pickle=True)
sat = result['arr_0'][()]
sat.label = "sat"
celestial_bodies = result['arr_1']

leg_1_data = np.load("leg_1_data.npy", allow_pickle=True)[()]
leg_2_data = np.load("leg_2_data.npy", allow_pickle=True)[()]
leg_3_data = np.load("leg_3_data.npy", allow_pickle=True)[()]

n_steps_1 = leg_1_data["n_steps"]
n_steps_2 = leg_2_data["n_steps"]
n_steps_3 = leg_3_data["n_steps"]
print(f'n_steps_1: {n_steps_1}')
print(f'n_steps_2: {n_steps_2}')
print(f'n_steps_3: {n_steps_3}')

# celestial_bodies = [sun, earth, moon, mars, mercury, jupiter, venus, saturn, uranus, neptune]
sun = celestial_bodies[0]
earth = celestial_bodies[1]
moon = celestial_bodies[2]
mars = celestial_bodies[3]
earth_orbit = leg_1_data["sat_orbit"]

"""
Plot LEG-1 Earth Depature
"""
start_index = 0
end_index = int(n_steps_1 + 20)
step = 3

if plot_leg_1:
    print("--------Generating Leg 1 Animation--------")

    plot_leg_1_timer = time.perf_counter()

    moon_wrt_earth = moon.r_ar[start_index:end_index:step,
                               :] - earth.r_ar[start_index:end_index:step, :]
    sat_wrt_earth = sat.r_ar[start_index:end_index:step,
                             :] - earth.r_ar[start_index:end_index:step, :]

    n_frames = len(sat_wrt_earth)
    azim_angle = np.linspace(45, 180+45, n_frames)

    metadata = dict(title='Leg1Ani', artist='jonamat03@gmail.com')
    writer = FFMpegWriter(fps=24, metadata=metadata)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    with writer.saving(fig, "leg_1_animation.mp4", 220):
        for i in range(n_frames):
            plt.cla()
            # Earth point
            ax.scatter(0, 0, 0, color=earth.color, s=20,
                       marker='o', edgecolor='k', linewidth=0.3, label=earth.label)
            # Sat
            ax.plot(sat_wrt_earth[:i, 0], sat_wrt_earth[:i, 1],
                    sat_wrt_earth[:i, 2], color=sat.color, label=sat.label)

            ax.scatter(sat_wrt_earth[i, 0], sat_wrt_earth[i, 1], sat_wrt_earth[i, 2],
                       color=sat.color, s=7, marker='o', edgecolor='k', linewidth=0.4)

            # Moon
            ax.plot(moon_wrt_earth[:i, 0], moon_wrt_earth[:i, 1],
                    moon_wrt_earth[:i, 2], color=moon.color, label=moon.label)

            ax.scatter(moon_wrt_earth[i, 0], moon_wrt_earth[i, 1], moon_wrt_earth[i, 2],
                       color=moon.color, s=20, marker='o', edgecolor='k', linewidth=0.3)

            ax.set_title(f"LEG-1: Earth Depature", fontsize=14, pad=10)
            ax.set_aspect('equal')
            ax.set_xlabel("X [km]")
            ax.set_ylabel("Y [km]")
            ax.set_zlabel("Z [km]")
            ax.set_xlim([25000, -400000])
            ax.set_ylim([25000, -200000])
            ax.set_zlim([25000, -100000])
            plt.tight_layout()

            ax.view_init(elev=45, azim=azim_angle[i], roll=0)
            ax.legend(loc='best')
            writer.grab_frame()
            print(f'Generated Leg 1 frame {i} of {n_frames}')

        # Pause on last frame for 5 seconds
        print("Generating End-Pause")
        for _ in range(24 * 5):  # 24 fps × 5 sec
            writer.grab_frame()

    print(
        f"Generating Leg 1 took {time.perf_counter() - plot_leg_1_timer} sec")


"""
Plot LEG-2 Transfer
"""
start_index = n_steps_1 - 50
end_index = n_steps_1 + n_steps_2 + 50
step = 2

if plot_leg_2:
    print("--------Generating Leg 2 Animation--------")
    plot_leg_2_timer = time.perf_counter()

    sat_wrt_sun = sat.r_ar[start_index:end_index:step, :] - \
        sun.r_ar[start_index:end_index:step, :]
    earth_wrt_sun = earth.r_ar[start_index:end_index:step,
                               :] - sun.r_ar[start_index:end_index:step, :]
    mars_wrt_sun = mars.r_ar[start_index:end_index:step,
                             :] - sun.r_ar[start_index:end_index:step, :]

    n_frames = len(sat_wrt_sun)
    azim_angle = np.linspace(0, -120, n_frames)

    metadata = dict(title='Leg2Ani', artist='jonamat03@gmail.com')
    writer = FFMpegWriter(fps=24, metadata=metadata)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    with writer.saving(fig, "leg_2_animation.mp4", 100):
        for i in range(n_frames):
            plt.cla()

            # SUN point
            ax.scatter(0, 0, 0, color=sun.color, s=20,
                       marker='o', edgecolor='k', linewidth=0.3, label=sun.label)

            # Sat
            ax.plot(sat_wrt_sun[:i, 0], sat_wrt_sun[:i, 1],
                    sat_wrt_sun[:i, 2], color=sat.color, label=sat.label)

            ax.scatter(sat_wrt_sun[i, 0], sat_wrt_sun[i, 1], sat_wrt_sun[i, 2],
                       color=sat.color, s=7, marker='o', edgecolor='k', linewidth=0.4)

            # Earth
            ax.plot(earth_wrt_sun[:i, 0], earth_wrt_sun[:i, 1],
                    earth_wrt_sun[:i, 2], color=earth.color, label=earth.label)

            ax.scatter(earth_wrt_sun[i, 0], earth_wrt_sun[i, 1], earth_wrt_sun[i, 2],
                       color=earth.color, s=7, marker='o', edgecolor='k', linewidth=0.4)

            # Mars
            ax.plot(mars_wrt_sun[:i, 0], mars_wrt_sun[:i, 1],
                    mars_wrt_sun[:i, 2], color=mars.color, label=mars.label)

            ax.scatter(mars_wrt_sun[i, 0], mars_wrt_sun[i, 1], mars_wrt_sun[i, 2],
                       color=mars.color, s=7, marker='o', edgecolor='k', linewidth=0.4)

            ax.set_title(f"LEG-2: Transfer", fontsize=14, pad=10)
            ax.set_aspect('equal')
            ax.set_xlabel("X [km]")
            ax.set_ylabel("Y [km]")
            ax.set_zlabel("Z [km]")
            plt.tight_layout()

            ax.view_init(elev=30, azim=azim_angle[i], roll=0)
            ax.legend(loc='best')
            writer.grab_frame()
            print(f'Generated Leg 2 frame {i} of {n_frames}')

        # Pause on last frame for 5 seconds
        print("Generating End-Pause")
        for _ in range(24 * 5):  # 24 fps × 5 sec
            writer.grab_frame()

    print(
        f"Generating Leg 2 took {time.perf_counter() - plot_leg_2_timer} sec")

"""
Plot LEG-3 Transfer
"""
start_index = n_steps_1 + n_steps_2 - 2
end_index = -int(n_steps_3/2)
step = 1

if plot_leg_3:
    print("--------Generating Leg 3 Animation--------")
    plot_leg_3_timer = time.perf_counter()

    sat_wrt_mars = sat.r_ar[start_index:end_index:step, :] - \
        mars.r_ar[start_index:end_index:step, :]

    n_frames = len(sat_wrt_mars)
    azim_angle = np.linspace(60, 180, n_frames)

    metadata = dict(title='Leg3Ani', artist='jonamat03@gmail.com')
    writer = FFMpegWriter(fps=24, metadata=metadata)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    with writer.saving(fig, "leg_3_animation.mp4", 220):
        for i in range(n_frames):
            plt.cla()

            # Mars point
            ax.scatter(0, 0, 0, color=mars.color, s=10,
                       marker='o', edgecolor='k', linewidth=0.3, label=mars.label)

            # Sat
            ax.plot(sat_wrt_mars[:i, 0], sat_wrt_mars[:i, 1],
                    sat_wrt_mars[:i, 2], color=sat.color, label=sat.label)

            ax.scatter(sat_wrt_mars[i, 0], sat_wrt_mars[i, 1], sat_wrt_mars[i, 2],
                       color=sat.color, s=7, marker='o', edgecolor='k', linewidth=0.4)

            ax.set_title(f"LEG-3: Mars Arrival", fontsize=14, pad=10)
            ax.set_aspect('equal')
            ax.set_xlabel("X [km]")
            ax.set_ylabel("Y [km]")
            ax.set_zlabel("Z [km]")
            ax.set_xlim([-4000, 4000])
            ax.set_ylim([-10000, 10000])
            ax.set_zlim([-4000, 4000])
            plt.tight_layout()

            ax.view_init(elev=30, azim=azim_angle[i], roll=0)
            ax.legend(loc='best')
            writer.grab_frame()
            print(f'Generated Leg 3 frame {i} of {n_frames}')

        # Pause on last frame for 5 seconds
        print("Generating End-Pause")
        for _ in range(24 * 5):  # 24 fps × 5 sec
            writer.grab_frame()

    print(
        f"Generating Leg 3 took {time.perf_counter() - plot_leg_3_timer} sec")
