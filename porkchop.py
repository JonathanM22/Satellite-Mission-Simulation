"""
porkchop.py Uses JPL ephemeris data as r1 and r2 to calculate the the lamberst problem over a range
of arrival and depature dates and generates a porkchop plot. 
"""
from orbit import *
import numpy as np
from scipy.integrate import ode
from scipy import optimize
import matplotlib.pyplot as plt

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
DV_CUT_OFF = 30*1000

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

# define range of depature & arrivial dates
depature_date_1 = Time("2020-06-01")
depature_date_2 = Time("2020-09-01")

arrival_date_1 = Time("2020-11-01")
arrival_date_2 = Time("2022-01-24")

"""
# define range of depature & arrivial dates
depature_date_1 = Time("2020-06-01")
depature_date_2 = Time("2020-06-07")

arrival_date_1 = Time("2022-06-01")
arrival_date_2 = Time("2022-09-01")
"""

step = TimeDelta(1, format='jd')

depature_dates = np.arange(
    depature_date_1, depature_date_2+step, step)

arrival_dates = np.arange(
    arrival_date_1, arrival_date_2+step, step)

delta_v_short = np.zeros([len(arrival_dates), len(depature_dates)])
delta_v_long = np.zeros([len(arrival_dates), len(depature_dates)])
tof_days = np.zeros([len(arrival_dates), len(depature_dates)])

for dd in np.arange(len(depature_dates)):
    for ar in np.arange(len(arrival_dates)):

        # calculate tof
        tof = (arrival_dates[ar] - depature_dates[dd]).sec  # type:ignore

        r1_eph, v1_eph = get_body_barycentric_posvel(
            'earth', depature_dates[dd])
        r2_eph, v2_eph = get_body_barycentric_posvel('mars', arrival_dates[ar])

        """
        Need to get sun position and velocity to transform
        baycentric cords to helio-centric
        """
        # Position of Sun
        r_sun1, v_sun1 = get_body_barycentric_posvel('sun', depature_dates[dd])
        r_sun2, v_sun2 = get_body_barycentric_posvel('sun', depature_dates[dd])

        # Position & Velocity of earth respect to sun
        r1 = (r1_eph.xyz - r_sun1.xyz).to(u.m).value  # type:ignore
        v1 = (v1_eph.xyz - v_sun1.xyz).to(u.m/u.s).value  # type:ignore

        # Position & Velocity of mars arrvial respect to sun
        r2 = (r2_eph.xyz - r_sun2.xyz).to(u.m).value  # type:ignore
        v2 = (v2_eph.xyz - v_sun2.xyz).to(u.m/u.s).value  # type:ignore

        #
        # Solve for transfer orbit via lamberts short path
        #
        try:
            transfer.a, transfer.p, transfer.e, transfer_v1, transfer_v2 = lambert_solver(
                r1, r2, tof, transfer.mu, desired_path='short')  # type:ignore

            transfer_r1 = r1

            # delta v1 is escaping earth
            delta_v1 = np.linalg.norm(transfer_v1 - v1)
            # delta v2 is for learving transfer orbit
            delta_v2 = np.linalg.norm(v2 - transfer_v2)
            # total delta v
            if (delta_v1 + delta_v2) > DV_CUT_OFF:
                delta_v_short[ar, dd] = DV_CUT_OFF
            else:
                delta_v_short[ar, dd] = delta_v1 + delta_v2
        except:
            delta_v_short[ar, dd] = DV_CUT_OFF

        #
        # Solve for transfer orbit via lamberts long path
        #
        try:
            transfer.a, transfer.p, transfer.e, transfer_v1, transfer_v2 = lambert_solver(
                r1, r2, tof, transfer.mu, desired_path='long')  # type:ignore

            transfer_r1 = r1

            # delta v1 is escaping earth
            delta_v1 = np.linalg.norm(transfer_v1 - v1)
            # delta v2 is for learving transfer orbit
            delta_v2 = np.linalg.norm(v2 - transfer_v2)
            # total delta v
            if (delta_v1 + delta_v2) > DV_CUT_OFF:
                delta_v_long[ar, dd] = DV_CUT_OFF
            else:
                delta_v_long[ar, dd] = delta_v1 + delta_v2
        except:
            delta_v_long[ar, dd] = DV_CUT_OFF

        tof_days[ar, dd] = (arrival_dates[ar] -
                            depature_dates[dd]).to('day').value  # type:ignore

        print(f'SHORT: {delta_v_short[ar, dd]} | LONG: {delta_v_long[ar, dd]}')

# Convert to km/s
delta_v_short /= 1000
delta_v_long /= 1000

"""- - - - - - - - - - - - - - - -PLOTTING- - - - - - - - - - - - - - - -"""
fig, ax = plt.subplots(figsize=(10, 20))
ft = 15

dv_levels = np.arange(0, (DV_CUT_OFF/1000)+5, 5)
cf = ax.contourf(delta_v_short, levels=dv_levels, cmap="viridis")
cbar = fig.colorbar(cf, ax=ax, label="ΔV [km/s]")

dv_short = ax.contour(delta_v_short,  levels=dv_levels, colors='k')
ax.clabel(dv_short, inline=True, fontsize=ft, fmt="%i")

dv_long = ax.contour(delta_v_long,  levels=dv_levels, colors='k')
ax.clabel(dv_long, inline=True, fontsize=ft, fmt="%i")

tof_levels = np.arange(np.min(tof_days), np.max(tof_days), 50)
tof = ax.contour(tof_days,  levels=tof_levels, colors='white')
ax.clabel(tof, inline=True, fontsize=ft, fmt="%i")

# formatting
ax.set_xlabel(
    f"Days after ({depature_date_1.to_datetime().strftime('%Y-%m-%d')})")
ax.set_ylabel(
    f"Days after ({arrival_date_1.to_datetime().strftime('%Y-%m-%d')})")
ax.set_title("Earth to Mars Porkchop Plot")
# ax.set_aspect('equal')
fig.savefig("porkchop_plot.pdf")
plt.show()


plot = False
if plot == True:
    dt = 86400
    """- - - - - - - - - - - - - - - -PLOTTING- - - - - - - - - - - - - - - -"""
    earth_rs, earth_vs = propogate_orbit(
        r1, v1, earth.mu, tspan=tof.sec, dt=dt)
    mars_rs, mars_vs = propogate_orbit(
        r1_mars, v1_mars, mars.mu, tspan=tof.sec, dt=dt)
    transfer_rs, transfer_vs = propogate_orbit(
        transfer_r1, transfer_v1, transfer.mu, tspan=tof.sec, dt=dt)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(earth_rs[:, 0], earth_rs[:, 1],
            earth_rs[:, 2], color='green', label='earth')
    ax.plot(mars_rs[:, 0], mars_rs[:, 1],
            mars_rs[:, 2], color='red', label='mars')
    ax.plot(transfer_rs[:, 0], transfer_rs[:, 1],
            transfer_rs[:, 2], color='orange', label='transfer')

    # Add Sun
    ax.scatter(r_sun.xyz.value[0], r_sun.xyz.value[1], r_sun.xyz.value[2],
               color='yellow', s=15, marker='o', edgecolor='k', label="SUN")

    # Add Earth Departure Point
    ax.scatter(earth_rs[0, 0], earth_rs[0, 1], earth_rs[0, 2],
               color='green', s=15, marker='o', edgecolor='k', label="Earth Depature")

    # Add Earth Arrival Point
    ax.scatter(earth_rs[-1, 0], earth_rs[-1, 1], earth_rs[-1, 2],
               color='green', s=15, marker='o', edgecolor='k', label="Earth Arrival")

    # Add Mars Departure Point
    ax.scatter(mars_rs[0, 0], mars_rs[0, 1], mars_rs[0, 2],
               color='red', s=15, marker='o', edgecolor='k', label="Mars Depature")

    # Add Mars Arrival Point
    ax.scatter(mars_rs[-1, 0], mars_rs[-1, 1], mars_rs[-1, 2],
               color='red', s=15, marker='o', edgecolor='k', label="Mars Arrival")

    # formatting
    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc='right')
    plt.show()
