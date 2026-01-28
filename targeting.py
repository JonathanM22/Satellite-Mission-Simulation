
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


# FROM VRAJ OPTIMAL C3
# C3 = 10.2
# Vinf_arrival = array([-2.3053362 ,  0.75875929,  0.93366132])
# RAAN_dep = 2.8236282989603887
# Dec_dep = 0.36724662932431584

# Define Earth Parking Orbit
earth_parking = Orbit(mu=EARTH_MU,
                      a=32000*u.km,
                      e=0.80*u.km/u.km,  # unitless
                      f0=(0*u.deg).to(u.rad),
                      inc=(28*u.deg).to(u.rad),
                      raan=(175*u.deg).to(u.rad),
                      aop=(240*u.deg).to(u.rad)
                      )


def v_to_raan_dec(v_inf):

    dec = np.arcsin(v_inf[2]/np.linalg.norm(v_inf))
    raan = np.arctan2(v_inf[1], v_inf[0])

    return raan, dec


def sat_orbit_targeting(orbit, v_inf_mag, x):
    # orbit: Orbit object

    orbit.raan = (x[0][0]*u.deg).to(u.rad)
    orbit.aop = (x[1][0]*u.deg).to(u.rad)

    r = orbit.r_at_true_anomaly(orbit.f0)
    v = np.sqrt(2*(orbit.energy + (orbit.mu/r)))

    v_hyp = np.sqrt(2*(((v_inf_mag**2)/2) + (orbit.mu/r)))

    r_pqw, v_pqw = orb_2_pqw(r.value,
                             orbit.f0.value, orbit.e.value,
                             orbit.p.value, orbit.mu.value)

    r_eci, v_eci = perif_2_eci(r_pqw, v_pqw, orbit.inc.value,
                               orbit.raan.value,
                               orbit.aop.value)

    sat_v_dir = v_eci/np.linalg.norm(v_eci)

    dv = v_hyp - v

    v_eci = (v_eci + dv.value*sat_v_dir)*(u.km/u.s)

    raan, dec = v_to_raan_dec(v_eci)

    return np.array([raan.value, dec.value]).reshape(2, 1)


def sensitivity_matrix(orbit, v_inf_mag, x, dt_raan, dt_aop):
    # Reshape dt_input args into dt vectors
    dt_rann_ar = np.array([dt_raan, 0]).reshape(2, 1)
    dt_aop_ar = np.array([0, dt_aop]).reshape(2, 1)

    dt_raan_col = (1/dt_raan)*(sat_orbit_targeting(
        orbit, v_inf_mag, x + dt_rann_ar) - sat_orbit_targeting(orbit, v_inf_mag, x))

    dt_aop_col = (1/dt_aop)*(sat_orbit_targeting(
        orbit, v_inf_mag, x + dt_aop_ar) - sat_orbit_targeting(orbit, v_inf_mag, x))

    return np.block([dt_raan_col, dt_aop_col])


# VRAJ OPTIMAL SOLN
C3 = 10.2
vinf_arr = np.array([-2.3053362,  0.75875929,  0.93366132])*(u.km/u.s)
vinf_mag = np.linalg.norm(vinf_arr).to(u.km/u.s)
vinf_raan, vinf_dec = v_to_raan_dec(vinf_arr)

raan0 = 370
aop0 = 54
x0 = np.array([raan0, aop0]).reshape(2, 1)
y0 = sat_orbit_targeting(earth_parking, vinf_mag, x0)
dt_raan = 10
dt_aop = 10

i = 0
max_i = 20000
tol = np.array([0.1, 0.1])
y_d = np.array([vinf_raan.value, vinf_dec.value]).reshape(2, 1)
x = x0
error = y0 - y_d

while np.linalg.norm(error) > 0.1:

    f_x = sat_orbit_targeting(earth_parking, vinf_mag, x)
    J = sensitivity_matrix(earth_parking, vinf_mag, x, dt_raan, dt_aop)

    x_k = x - J@(f_x-y_d)
    f_xk = sat_orbit_targeting(earth_parking, vinf_mag, x_k)
    error = (f_xk-y_d)

    print(f"[{i}] ERROR:{error.flatten()}")

    x = x_k
    i += 1
    if i > max_i:
        print(f"[MAX ITER] ERROR:{error.flatten()}")
        break

if np.linalg.norm(error) < 0.1:
    print(f"[TOL SATISFIED] ERROR:{error.flatten()}")
else:
    print(f"[TOL NOT SATISFIED] ERROR:{error.flatten()}")

f_x = sat_orbit_targeting(earth_parking, vinf_mag, x)
error = (f_x-y_d)
print(f"x: earth.raan = {x[0][0]}, earth.aop = {x[1][0]}")
print(f"SatVel@f| raan: {f_x[0][0]} rad | dec: {f_x[1][0]} rad")
print(f"V_inf| raan: {y_d[0][0]} rad | dec: {y_d[1][0]} rad")
