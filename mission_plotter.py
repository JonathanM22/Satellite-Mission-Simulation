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

# Plotting options
plot_leg_1 = False
plot_leg_2 = False
plot_leg_3 = True
sat_mission = False

"""
Plot LEG-1
"""
leg_1_data = np.load("leg_1_data.npy", allow_pickle=True)[()]

central_body = leg_1_data["central_body"]
sat_orbit = leg_1_data["sat_orbit"]
bodies = leg_1_data["bodies"]
dt = leg_1_data["dt"]
n_steps_1 = leg_1_data["n_steps"]
t0 = leg_1_data["t0"]
tf = leg_1_data["tf"]
y0 = leg_1_data["y0"]
ts = leg_1_data["ts"]
ys = leg_1_data["ys"]

if plot_leg_1:

    # Filling pos and vel data for all celestial bodies
    central_body.r_ar = np.zeros((len(ts), 3))
    central_body.v_ar = np.zeros((len(ts), 3))
    for i, t in enumerate(ts):
        r, v = get_body_barycentric_posvel(central_body.label, t)

        central_body.r_ar[i] = r.xyz.to(u.km).value
        central_body.v_ar[i] = v.xyz.to(u.km/u.s).value

    for body in bodies:
        body.r_ar = np.zeros((len(ts), 3))
        body.v_ar = np.zeros((len(ts), 3))
        for i, t in enumerate(ts):
            r, v = get_body_barycentric_posvel(body.label, t)

            body.r_ar[i] = r.xyz.to(u.km).value
            body.v_ar[i] = v.xyz.to(u.km/u.s).value

    ax = plt.figure().add_subplot(projection='3d')

    # Plot central body
    ax.scatter(0, 0, 0, color=central_body.color, s=5,
               marker='o', edgecolor='k', label=central_body.label)

    # Plot all bodies around central body
    for body in bodies:
        body_wrt_earth = body.r_ar - central_body.r_ar

        ax.plot(body_wrt_earth[:, 0],
                body_wrt_earth[:, 1],
                body_wrt_earth[:, 2],
                color=body.color,
                label=body.label)

    ax.plot(ys[:, 0], ys[:, 1], ys[:, 2],
            color=sat.color,
            label=sat.label)

    ax.set_title(f"LEG-1: {central_body.label} Frame", fontsize=14, pad=10)
    ax.set_aspect('equal')
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    plt.tight_layout()
    # Size limit to only see sat orbit
    # ax.set_xlim([-50000, 50000])
    # ax.set_ylim([-50000, 50000])
    # ax.set_zlim([-50000, 50000])
    ax.view_init(elev=np.rad2deg(sat_orbit.inc.value),
                 azim=90, roll=0)
    plt.show()

"""
Plot LEG-2
"""
leg_2_data = np.load("leg_2_data.npy", allow_pickle=True)[()]

central_body = leg_2_data["central_body"]
sat_orbit = leg_2_data["sat_orbit"]
bodies = leg_2_data["bodies"]
dt = leg_2_data["dt"]
n_steps_2 = leg_2_data["n_steps"]
t0 = leg_2_data["t0"]
tf = leg_2_data["tf"]
y0 = leg_2_data["y0"]
ts = leg_2_data["ts"]
ys = leg_2_data["ys"]

if plot_leg_2:

    # Filling pos and vel data for all celestial bodies
    central_body.r_ar = np.zeros((len(ts), 3))
    central_body.v_ar = np.zeros((len(ts), 3))
    for i, t in enumerate(ts):
        r, v = get_body_barycentric_posvel(central_body.label, t)

        central_body.r_ar[i] = r.xyz.to(u.km).value
        central_body.v_ar[i] = v.xyz.to(u.km/u.s).value

    for body in bodies:
        body.r_ar = np.zeros((len(ts), 3))
        body.v_ar = np.zeros((len(ts), 3))
        for i, t in enumerate(ts):
            r, v = get_body_barycentric_posvel(body.label, t)

            body.r_ar[i] = r.xyz.to(u.km).value
            body.v_ar[i] = v.xyz.to(u.km/u.s).value

    ax = plt.figure().add_subplot(projection='3d')

    # Plot central body
    ax.scatter(0, 0, 0, color=central_body.color, s=5,
               marker='o', edgecolor='k', label=central_body.label)

    # Plot all bodies around central body
    for body in bodies:
        body_wrt_earth = body.r_ar - central_body.r_ar

        ax.plot(body_wrt_earth[:, 0],
                body_wrt_earth[:, 1],
                body_wrt_earth[:, 2],
                color=body.color,
                label=body.label)

    ax.plot(ys[:, 0], ys[:, 1], ys[:, 2],
            color=sat.color,
            label=sat.label)

    ax.set_title(f"LEG-2: {central_body.label} Frame", fontsize=14, pad=10)
    ax.set_aspect('equal')
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    plt.tight_layout()
    # Size limit to only see sat orbit
    # ax.set_xlim([-50000, 50000])
    # ax.set_ylim([-50000, 50000])
    # ax.set_zlim([-50000, 50000])
    ax.view_init(elev=90, azim=90, roll=0)
    plt.show()

"""
Plot LEG-3
"""
leg_3_data = np.load("leg_3_data.npy", allow_pickle=True)[()]

central_body = leg_3_data["central_body"]
sat_orbit = leg_3_data["sat_orbit"]
bodies = leg_3_data["bodies"]
dt = leg_3_data["dt"]
n_steps_3 = leg_3_data["n_steps"]
t0 = leg_3_data["t0"]
tf = leg_3_data["tf"]
y0 = leg_3_data["y0"]
ts = leg_3_data["ts"]
ys = leg_3_data["ys"]


if plot_leg_3:

    # Filling pos and vel data for all celestial bodies
    central_body.r_ar = np.zeros((len(ts), 3))
    central_body.v_ar = np.zeros((len(ts), 3))
    for i, t in enumerate(ts):
        r, v = get_body_barycentric_posvel(central_body.label, t)

        central_body.r_ar[i] = r.xyz.to(u.km).value
        central_body.v_ar[i] = v.xyz.to(u.km/u.s).value

    for body in bodies:
        body.r_ar = np.zeros((len(ts), 3))
        body.v_ar = np.zeros((len(ts), 3))
        for i, t in enumerate(ts):
            r, v = get_body_barycentric_posvel(body.label, t)

            body.r_ar[i] = r.xyz.to(u.km).value
            body.v_ar[i] = v.xyz.to(u.km/u.s).value

    ax = plt.figure().add_subplot(projection='3d')

    # Plot central body
    ax.scatter(0, 0, 0, color=central_body.color, s=5,
               marker='o', edgecolor='k', label=central_body.label)

    # Plot all bodies around central body
    for body in bodies:
        body_wrt_earth = body.r_ar - central_body.r_ar

        ax.plot(body_wrt_earth[:, 0],
                body_wrt_earth[:, 1],
                body_wrt_earth[:, 2],
                color=body.color,
                label=body.label)

    ax.plot(ys[:, 0], ys[:, 1], ys[:, 2],
            color=sat.color,
            label=sat.label)

    ax.set_title(f"LEG-3: {central_body.label} Frame", fontsize=14, pad=10)
    ax.set_aspect('equal')
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    plt.tight_layout()
    # Size limit to only see sat orbit
    # ax.set_xlim([-50000, 50000])
    # ax.set_ylim([-50000, 50000])
    # ax.set_zlim([-50000, 50000])
    ax.view_init(elev=0, azim=0, roll=0)
    plt.show()

"""
Plot Sat mission
"""
# celestial_bodies = [sun, earth, moon, mars, mercury, jupiter, venus, saturn, uranus, neptune]'
sun = celestial_bodies[0]
earth = celestial_bodies[1]
mars = celestial_bodies[3]
mission_bodies = [mars]

start_index = n_steps_1 + n_steps_2
end_index = -1

if sat_mission:
    ts = sat.t_ar
    rs = sat.r_ar
    vs = sat.v_ar
    # Filling pos and vel data for all celestial bodies
    # Relative to bary center
    for mission_body in mission_bodies:
        mission_body.r_ar = np.zeros((len(ts), 3))
        mission_body.v_ar = np.zeros((len(ts), 3))
        for i, t in enumerate(ts):
            r, v = get_body_barycentric_posvel(mission_body.label, t)

            mission_body.r_ar[i] = r.xyz.to(u.km).value
            mission_body.v_ar[i] = v.xyz.to(u.km/u.s).value

    ax = plt.figure().add_subplot(projection='3d')

    for mission_body in mission_bodies:
        ax.plot(mission_body.r_ar[start_index:end_index, 0],
                mission_body.r_ar[start_index:end_index, 1],
                mission_body.r_ar[start_index:end_index, 2],
                color=mission_body.color,
                label=mission_body.label)

    ax.plot(rs[start_index:end_index, 0], rs[start_index:end_index, 1], rs[start_index:end_index, 2],
            color=sat.color,
            label=sat.label)

    ax.set_title(f"MISSION", fontsize=14, pad=10)
    ax.set_aspect('equal')
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    plt.show()
