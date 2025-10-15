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
DV_CUT_OFF = 30*1000  # km/s

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

# define range of depature & arrivial dates
"""
depature_date_1 = Time("2020-07-01")
depature_date_2 = Time("2020-09-01")

arrival_date_1 = Time("2020-11-01")
arrival_date_2 = Time("2022-11-10")
"""

# define range of depature & arrivial dates
depature_date_1 = Time("2026-10-01")
depature_date_2 = Time("2027-01-30")

arrival_date_1 = Time("2027-08-01")
arrival_date_2 = Time("2028-02-28")


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

        r1_earth_eph, v1_earth_eph = get_body_barycentric_posvel(
            'earth', depature_dates[dd])
        r2_mars_eph, v2_mars_eph = get_body_barycentric_posvel(
            'mars', depature_dates[dd])

        """
        Need to get sun position and velocity to transform
        baycentric cords to helio-centric
        """
        # Position of Sun
        r1_sun, v1_sun = get_body_barycentric_posvel('sun', depature_dates[dd])

        # Position & Velocity of earth respect to sun @ Depature
        r1 = (r1_earth_eph.xyz - r1_sun.xyz).to(u.m).value  # type:ignore
        v1 = (v1_earth_eph.xyz - v1_sun.xyz).to(u.m/u.s).value  # type:ignore

        # Position & Velocity of mars respect to sun @ Depature
        r1_mars = (r2_mars_eph.xyz - r1_sun.xyz).to(u.m).value  # type:ignore
        v1_mars = (v2_mars_eph.xyz - v1_sun.xyz).to(u.m /       # type:ignore
                                                    u.s).value  # type:ignore

        """
        Propogate from JPL data to get the arrivial position of the bodies. 
        """
        dt = 86400
        earth_rs, earth_vs = propogate_orbit(
            r1, v1, earth.mu, tspan=tof, dt=dt)
        mars_rs, mars_vs = propogate_orbit(
            r1_mars, v1_mars, mars.mu, tspan=tof, dt=dt)

        r2_mars = mars_rs[-1]
        v2_mars = mars_vs[-1]

        """
        Solve for transfer orbit via lamberts SHORT path
        """
        try:
            transfer.a, transfer.p, transfer.e, transfer_v1, transfer_v2 = lambert_solver(
                r1, r2_mars, tof, transfer.mu, desired_path='short')  # type:ignore

            transfer_r1 = r1

            # delta v1 is escaping earth
            delta_v1 = np.linalg.norm(transfer_v1 - v1)

            # delta v2 is for learving transfer orbit
            delta_v2 = np.linalg.norm(v2_mars - transfer_v2)

            # total delta v
            if (delta_v1 + delta_v2) > DV_CUT_OFF:
                delta_v_short[ar, dd] = DV_CUT_OFF
            else:
                delta_v_short[ar, dd] = delta_v1 + delta_v2
        except:
            delta_v_short[ar, dd] = DV_CUT_OFF

        """
        Solve for transfer orbit via lamberts LONG path
        """
        try:
            transfer.a, transfer.p, transfer.e, transfer_v1, transfer_v2 = lambert_solver(
                r1, r2_mars, tof, transfer.mu, desired_path='long')  # type:ignore

            transfer_r1 = r1

            # delta v1 is escaping earth
            delta_v1 = np.linalg.norm(transfer_v1 - v1)

            # delta v2 is for learving transfer orbit
            delta_v2 = np.linalg.norm(v2_mars - transfer_v2)
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
