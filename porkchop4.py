"""
porkchop4.py a imporvment on 3, with the goal of adding multiprocessing to reduce compute time
I just made a seperate file to compare with porkchop3.py

Uses JPL ephemeris data as r1 and r2 to calculate the the lamberst problem over a range
of arrival and depature dates and generates a porkchop plot.
"""
from orbit import *
import numpy as np
from scipy.integrate import ode
from scipy import optimize
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from orbit_util import *
from astropy.time import Time
from astropy.time import TimeDelta
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel

# Constants
# ALL constants are in SI UNITS! (meters, seconds, etc.)
# Also for formattng constants are ALL_CAPS
EARTH_RAD = 6.371 * 10**6
AU = 1.496 * 10**11
SUN_MU = 1.327 * 10**20
DV_CUT_OFF = 150*1000  # km/s


def porkchop_point(date_info):
    """
    arrivial_date, depature_date: Astropy Time Object
    returns: delta_v_short[m/s], delta_v_long[m/s], tof_days
    """

    i, j, depature_date, arrival_date = date_info

    # calculate tof
    tof = (arrival_date - depature_date).sec  # type:ignore

    """
    Need to get sun position and velocity to transform
    baycentric cords to helio-centric
    """

    # Position of Sun @ Depature and Arrival
    r1_sun, v1_sun = get_body_barycentric_posvel('sun', depature_date)
    r2_sun, v2_sun = get_body_barycentric_posvel('sun', arrival_date)

    # Position & Velocity of EARTH respect to sun @ Depature
    r1_earth_eph, v1_earth_eph = get_body_barycentric_posvel(
        'earth', depature_date)
    r1_earth = (r1_earth_eph.xyz - r1_sun.xyz).to(u.m).value      # type:ignore
    v1_earth = (v1_earth_eph.xyz - v1_sun.xyz).to(u.m/u.s).value  # type:ignore

    # Position & Velocity of MARS respect to sun @ Arrivial
    r2_mars_eph, v2_mars_eph = get_body_barycentric_posvel(
        'mars', arrival_date)
    r2_mars = (r2_mars_eph.xyz - r2_sun.xyz).to(u.m).value      # type:ignore
    v2_mars = (v2_mars_eph.xyz - v2_sun.xyz).to(u.m/u.s).value  # type:ignore

   # Solve for lamberts the short way
    _, _, _, transfer_v1, transfer_v2 = result = lambert_solver(
        r1_earth, r2_mars, tof, SUN_MU, desired_path='short')  # type:ignore

    # delta v1 is escaping earth
    delta_v1 = np.linalg.norm(transfer_v1 - v1_earth)

    # delta v2 is for learving transfer orbit
    delta_v2 = np.linalg.norm(v2_mars - transfer_v2)

    # total delta v
    if (delta_v1 + delta_v2) > DV_CUT_OFF:
        delta_v_short = DV_CUT_OFF
    else:
        delta_v_short = delta_v1 + delta_v2

   # Solve for lamberts the long way
    _, _, _, transfer_v1, transfer_v2 = lambert_solver(
        r1_earth, r2_mars, tof, SUN_MU, desired_path='long')  # type:ignore

    # delta v1 is escaping earth
    delta_v1 = np.linalg.norm(transfer_v1 - v1_earth)

    # delta v2 is for learving transfer orbit
    delta_v2 = np.linalg.norm(v2_mars - transfer_v2)
    # total delta v
    if (delta_v1 + delta_v2) > DV_CUT_OFF:
        delta_v_long = DV_CUT_OFF
    else:
        delta_v_long = delta_v1 + delta_v2

    tof_days = (arrival_date - depature_date).to('day').value  # type:ignore

    return i, j, delta_v_short, delta_v_long, tof_days


def main():
    """
    Orbital elemenets are not used in this code, but we use the orbit class
    """
    transfer = Orbit(mu=SUN_MU)

    # intialize jpl ephemeris
    solar_system_ephemeris.set('de432s')  # Ephemeris from 1950 - 2050

    # Start run timer
    start = time.perf_counter()

    """
    # define range of depature & arrivial dates
    depature_date_1 = Time("2020-07-01")
    depature_date_2 = Time("2020-08-10")

    arrival_date_1 = Time("2022-11-01")
    arrival_date_2 = Time("2022-12-10")
    """

    # define range of depature & arrivial dates
    depature_date_1 = Time("2026-10-01")
    depature_date_2 = Time("2027-01-30")

    arrival_date_1 = Time("2027-08-01")
    arrival_date_2 = Time("2028-02-28")

    # Create array of date pairs
    step = TimeDelta(1, format='jd')
    depature_dates = np.arange(
        depature_date_1, depature_date_2+step, step)

    arrival_dates = np.arange(
        arrival_date_1, arrival_date_2+step, step)

    date_info = np.zeros(
        [len(depature_dates), len(arrival_dates)], dtype=object)

    for i, dep in enumerate(depature_dates):
        for j, arv in enumerate(arrival_dates):
            date_info[i, j] = (i, j, dep, arv)

    # Intilize result arrays
    delta_v_short_arr = np.zeros(date_info.shape)
    delta_v_long_arr = np.zeros(date_info.shape)
    tof_days_arr = np.zeros(date_info.shape)

    with Pool(10) as pool:
        results = pool.map(
            porkchop_point, date_info.flatten())

        for (i, j, delta_v_short, delta_v_long, tof_days) in results:
            delta_v_short_arr[i, j] = delta_v_short
            delta_v_long_arr[i, j] = delta_v_long
            tof_days_arr[i, j] = tof_days

    # END run timer
    end = time.perf_counter()

    print(f" TIME TO RUN PROGRAM IS {end-start}")

    # Convert to km/s
    delta_v_short_arr /= 1000
    delta_v_long_arr /= 1000

    """- - - - - - - - - - - - - - - -PLOTTING- - - - - - - - - - - - - - - -"""
    fig, ax = plt.subplots(figsize=(8, 16))
    ft = 12

    dv_levels = np.arange(0, (DV_CUT_OFF/1000), 3)

    delta_v_best = np.minimum(delta_v_short_arr, delta_v_long_arr)

    cf = ax.contourf(delta_v_best, inline=False, levels=dv_levels)
    cbar = fig.colorbar(cf, ax=ax, label="ΔV [km/s]")

    tof_levels = np.arange(np.min(tof_days_arr), np.max(tof_days_arr), 50)
    tof = ax.contour(tof_days_arr, levels=tof_levels, colors='black')
    ax.clabel(tof, inline=True, fontsize=ft, fmt="%i")

    # formatting
    ax.set_xlabel(
        # type:ignore
        f"Depature Julian Date ({depature_date_1.to_datetime().strftime('%Y-%m-%d')})")
    ax.set_ylabel(
        # type:ignore
        f"Arrivial Julian Date ({arrival_date_1.to_datetime().strftime('%Y-%m-%d')})")
    ax.set_title("Earth to Mars Porkchop Plot")

    fig.savefig("porkchop_plot_2.png")
    # plt.show()


if __name__ == "__main__":
    main()
