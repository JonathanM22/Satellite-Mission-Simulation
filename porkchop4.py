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


def porkchop_point(date_pair, transfer):
    """
    arrivial_date, depature_date: Astropy Time Object
    returns: delta_v_short[m/s], delta_v_long[m/s], tof_days
    """

    depature_date, arrival_date = date_pair

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

    """
    Solve for transfer orbit via lamberts SHORT path
    """
    try:
        transfer.a, transfer.p, transfer.e, transfer_v1, transfer_v2 = lambert_solver(
            r1_earth, r2_mars, tof, transfer.mu, desired_path='short')  # type:ignore

        transfer_r1 = r1_earth

        # delta v1 is escaping earth
        delta_v1 = np.linalg.norm(transfer_v1 - v1_earth)

        # delta v2 is for learving transfer orbit
        delta_v2 = np.linalg.norm(v2_mars - transfer_v2)

        # total delta v
        if (delta_v1 + delta_v2) > DV_CUT_OFF:
            delta_v_short = DV_CUT_OFF
        else:
            delta_v_short = delta_v1 + delta_v2
    except:
        delta_v_short = DV_CUT_OFF

    """
    Solve for transfer orbit via lamberts LONG path
    """
    try:
        transfer.a, transfer.p, transfer.e, transfer_v1, transfer_v2 = lambert_solver(
            r1_earth, r2_mars, tof, transfer.mu, desired_path='long')  # type:ignore

        transfer_r1 = r1_earth

        # delta v1 is escaping earth
        delta_v1 = np.linalg.norm(transfer_v1 - v1_earth)

        # delta v2 is for learving transfer orbit
        delta_v2 = np.linalg.norm(v2_mars - transfer_v2)
        # total delta v
        if (delta_v1 + delta_v2) > DV_CUT_OFF:
            delta_v_long = DV_CUT_OFF
        else:
            delta_v_long = delta_v1 + delta_v2
    except:
        delta_v_long = DV_CUT_OFF

    tof_days = (arrival_date - depature_date).to('day').value  # type:ignore

    return delta_v_short, delta_v_long, tof_days


def main():
    """
    Orbital elemenets are not used in this code, but we use the orbit class
    """
    # Use Table 1 https://ssd.jpl.nasa.gov/planets/approx_pos.html
    earth = Orbit(a=1.00000261*AU,
                  e=0.01671123,
                  inc=-0.00001531,
                  raan=0,
                  aop=102.93768193,
                  mu=SUN_MU)

    mars = Orbit(a=1.523*AU,
                 e=0.093,
                 inc=1.849,
                 raan=49.5,
                 aop=-23.9,
                 mu=SUN_MU)

    transfer = Orbit(mu=SUN_MU)

    # intialize jpl ephemeris
    solar_system_ephemeris.set('de432s')  # Ephemeris from 1950 - 2050

    # Start run timer
    start = time.perf_counter()

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
    """

    # Create array of date pairs
    step = TimeDelta(1, format='jd')
    depature_dates = np.arange(
        depature_date_1, depature_date_2+step, step)

    arrival_dates = np.arange(
        arrival_date_1, arrival_date_2+step, step)

    date_pairs = np.zeros(
        [len(depature_dates), len(arrival_dates)], dtype=object)
    for dep in range(len(depature_dates)):
        for arv in range(len(arrival_dates)):
            date_pairs[dep, arv] = (depature_dates[dep], arrival_dates[arv])

    # Intilize result arrays
    delta_v_short = np.zeros(date_pairs.shape)
    delta_v_long = np.zeros(date_pairs.shape)
    tof_days = np.zeros(date_pairs.shape)

    flat_pairs = date_pairs.flatten()

    with Pool(10) as pool:
        results = pool.map(porkchop_point, flat_pairs)

    results = np.array(results)
    delta_v_short = results[:, 0].reshape(
        len(depature_dates), len(arrival_dates))
    delta_v_long = results[:, 1].reshape(
        len(depature_dates), len(arrival_dates))
    tof_days = results[:, 2].reshape(len(depature_dates), len(arrival_dates))

    # END run timer
    end = time.perf_counter()

    print(f" TIME TO RUN PROGRAM IS {end-start}")

    # Convert to km/s
    delta_v_short /= 1000
    delta_v_long /= 1000

    """- - - - - - - - - - - - - - - -PLOTTING- - - - - - - - - - - - - - - -"""
    fig, ax = plt.subplots(figsize=(8, 16))
    ft = 12

    dv_levels = np.arange(0, (DV_CUT_OFF/1000), 3)

    delta_v_best = np.minimum(delta_v_short, delta_v_long)

    cf = ax.contourf(delta_v_best, inline=False, levels=dv_levels)
    cbar = fig.colorbar(cf, ax=ax, label="ΔV [km/s]")

    # dv_short = ax.contour(delta_v_short,  inline=False,
    #                      levels=dv_levels, colors='k')
    # ax.clabel(dv_short, inline=False)

    # dv_long = ax.contour(delta_v_long,  inline=False, levels=dv_levels, colors='k')
    # ax.clabel(dv_long, inline=False)

    tof_levels = np.arange(np.min(tof_days), np.max(tof_days), 50)
    tof = ax.contour(tof_days, levels=tof_levels, colors='black')
    ax.clabel(tof, inline=True, fontsize=ft, fmt="%i")

    # formatting
    ax.set_xlabel(
        f"Depature Julian Date ({depature_date_1.to_datetime().strftime('%Y-%m-%d')})")
    ax.set_ylabel(
        f"Arrivial Julian Date ({arrival_date_1.to_datetime().strftime('%Y-%m-%d')})")
    ax.set_title("Earth to Mars Porkchop Plot")

    fig.savefig("porkchop_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
