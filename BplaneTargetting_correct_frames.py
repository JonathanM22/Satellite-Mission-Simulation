"""
n-body. Trying to do n-body propagation
"""

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
from poliastro.bodies import Sun
from poliastro.iod import vallado

import orbit
k = Sun.k
"""
Functions
"""

# -------------------------------------------------------------------------------------------------------------FUNCTIONS-----------------------------------------------------------------------------------------------------------------

# Sets up single Runge Kutta 4 Step
def RK4_single_step(fun, dt, t0, y0, fun_arg: list):

    # evaluates inputted function, fun, at t0, y0, and inputted args to create 4 constants to solve 1 rk4 step
    # inputted function name --> y_dot_n_ephemeris
    k1 = fun(t0, y0, fun_arg)
    k2 = fun((t0 + (dt/2)), (y0 + ((dt.value/2)*k1)), fun_arg)
    k3 = fun((t0 + (dt/2)), (y0 + ((dt.value/2)*k2)), fun_arg)
    k4 = fun((t0 + dt), (y0 + (dt.value*k3)), fun_arg)
    y1 = y0 + (dt.value/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y1

def propagate_rk4(r0, v0, t0, tf, dt, fun_arg: list):
    # time array equally by dt
    ts = np.arange(t0, tf, dt)
    n_steps = len(ts)
    ys = np.zeros((n_steps, 6))
    y0 = np.concatenate((r0, v0.flatten()))
    ys[0] = y0
    step = 1
    for i in range(n_steps - 1):
        ys[i+1] = RK4_single_step(y_dot_n_ephemeris,
                                  dt, ts[i], ys[i], fun_arg=fun_arg)
        step += 1
    r = ys[:, :3]
    v = ys[:, 3:6]

    return r, v, ys

def perif_2_eci_DCM( inc, raan, aop):
    """
    Transforms perifocal fram to ECI frame
    """
    # Rotation matrices
    R1 = np.array([  # Third axis rotation about raan
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan),  np.cos(raan), 0],
        [0,             0,            1]
    ])
    R2 = np.array([  # First axis rotation about inc
        [1, 0,              0],
        [0, np.cos(inc), -np.sin(inc)],
        [0, np.sin(inc),  np.cos(inc)]
    ])
    R3 = np.array([  # Third axis rotation about aop
        [np.cos(aop), -np.sin(aop), 0],
        [np.sin(aop),  np.cos(aop), 0],
        [0,            0,           1]
    ])
    perif_2_eci_DCM = R1 @ R2 @ R3

    return perif_2_eci_DCM 

# Two body motion ODE: creating the y_dot function for n-body with ephemeris data
def y_dot_n_ephemeris(t, y, fun_arg: list):
    """
        t: astropy time object
        y: np.array
        fun_arg: premade  

        direction matters!
        r = distance from central body -> sat
        r_c = distance from origin -> central body
        r_k = distance from origin -> kth body
        r_sk = distance from sat -> kth body
        r_s = distance from barycenter -> sat
        m_c = central body mass
        m_k = kth body mass
        r_ck = distance from central body -> kth body
    """
    central_body = fun_arg[0]
    bodies = fun_arg[1]

    # measuring all distances and velocties with respect to the central body
    # r = distance from central body -> sat
    r = y[0:3]
    r_mag = np.linalg.norm(r)
    v = y[3:6]

    # r_c = distance from origin -> central body
    r_c = get_body_barycentric(central_body.label, t).xyz.to(u.km).value
    m_c = central_body.mass.value

    central_body.mu = G.value * m_c
    # acceleration on satellite due to central body
    a = ((central_body.mu)/(r_mag**3)) * -r
    # print(f'Accel from CB: {a}')

    for body in bodies:

        #  # r_s = r_c + r
        # r_sk = r_k - r_s
        # r_k = distance from origin -> kth body

        r_k = get_body_barycentric(body.label, t).xyz.to(u.km).value
        m_k = body.mass.value

        # r_s = distance from barycenter -> sat
        # r_sk = distance from sat -> kth body
        # r_ck = distance from central body -> kth body

        r_ck = r_k - r_c
        r_sk = r_ck - r
        r_sk_mag = np.linalg.norm(r_sk)

        body.mu = G.value * m_k
        # acceleration on satellite due to kth bodies
        a_k = ((body.mu)/(r_sk_mag**3)) * r_sk
        # print(f'Accel from {body.label}: {a_k}')

        # acceleration on CB due to kth body 
        a_cb_k = ((body.mu)/(np.linalg.norm(r_ck)**3)) * r_ck

        # if r and v were wrt to barycenter, we would just have a = a + a_k ( a_cb_k term only arises since we are wrt central body, which is also being accelerated by other bodies)
        # total acceleration on satellite due to all bodies
        a = a + a_k - a_cb_k

        '''
        
        Since the Sun is being accelerated by other bodies (Jupiter especially), the Sun-centered frame is non-inertial.

        Newtons laws are only valid in inertial frames of reference. Since the central body is being accelerated by other bodies, it is not an inertial frame.
        Therefore, we have to account for the acceleration of the central body due to the other bodies when calculating the acceleration of the satellite.
        This is done by subtracting the acceleration of the central body due to the kth body from the acceleration of the satellite due to the kth body.
        This ensures that the acceleration of the satellite is calculated in an inertial frame of reference.

        we are "correcting" for the non inertial frame of reference ( the sun ) by subtracting its acceleration due to other bodies from the satellites acceleration

        WE AREN'T SAYING THE SUN IS INERTIAL. WE ARE JUST MAKING THE EQUATIONS WORK BY ACCOUNTING FOR ITS ACCELERATION.

        SINCE WE WANT THE SATS MOTION WRT THE SUN, WE HAVE TO CORRECT FOR THE SUNS NON INTERIAL ACCELERAITON

        WE AREN'T DOING A FRAME TRANSFORMATION. WE ARE DOING RELATIVE MOTION FORUMUATION. aS LONG AS ALL ACCEL IS CALC'D IN THE INERTIAL FRAME AND WE SUBTRACT THE CENTRAL BODY ACC, THE REL MOTION IS VALID

        *** - IF OUR SAT IS WRT TO BODY THAT IS ACC, MUST SUBTRACT ACC. THE SUN BEING NON INERTIAL IS FINE AS LONG AS WE ACCOUNT FOR IT: THE MOTION IS STILL HELIOCENTRIC, JUST RELATIVE TO THE ACCELERATING SUN 

        SUN CENTERED FRAME IS NON INERTIAL, BUT WE ARE ACCOUNTING FOR IT BY SUBTRACTING THE SUNS ACCELERATION DUE TO OTHER BODIES FROM THE SATELLITES ACCELERATION DUE TO THOSE BODIES
        
        WE'RE WORKING WITH A RELATIVE SUN CENTERED FRAME. By correcting for the Suns acceleration, the relative equations of motion are equivalent to those written in an inertial frame.”

        '''

    y_dot = np.concatenate((v, a))

    return y_dot

def y_dot_2body_earth(t, y, mu):
    r = y[:3]
    v = y[3:]
    r_mag = np.linalg.norm(r)
    a = -mu * r / r_mag**3
    return np.concatenate((v, a))

def propagate_to_vinf(r0, v0, mu_earth, dt, r_stop=9250000):
    y = np.concatenate((r0, v0))
    t = 0
    while True:
        y = RK4_single_step(
            y_dot_2body_earth,
            TimeDelta(dt, format='sec'),
            t,
            y,
            mu_earth
        )
        r = y[:3]
        if np.linalg.norm(r) > r_stop:
            return y[3:]   # velocity ≈ vinf
        t += dt

"""
Constants and Intialization
"""
# program_start_timer = time.perf_counter()
print("\n------------------------------------------------------------------------------------------------Started Simulation------------------------------------------------------------------------------------------------n")

# Bodies to save for entire mission plotter
celestial_bodies = [sun, earth, moon, mars, mercury, jupiter, venus, saturn, uranus, neptune]

# Intialize SAT
SAT_MASS = 100*u.kg
departure_date = Time("2026-10-19")
sat = Spacecraft(SAT_MASS, departure_date, label="sat", color="purple")

'''
---------------------------------------------------------------------------------------------------------------STEP 1--------------=------------------------------------------------------------------------------------------------

Start with Lambert's problem with R1/R2/TOF.  Pull R1 and R2 from and ephemeris file for Earth and Mars.
Iterate on Lambert's problem until you have chosen a solution that you are happy with (e.g. minimum C3 and arrival Vinfinity at Mars).

'''

transfer_short = Orbit(mu=SUN_MU)
transfer_long = Orbit(mu=SUN_MU)


'''
EXPERIMENTING WITH SYNODIC PERIODS
'''

# eqn found online for synodic period between two planets
# Synodic period = 1 / |(1/T1) - (1/T2)|
earth_period = 2*math.pi * np.sqrt((1.496e8**3)/SUN_MU)  # in seconds
mars_period = 2*math.pi * np.sqrt((2.279e8**3)/SUN_MU)  # in seconds
synodic_period_days = TimeDelta(1 / abs((1 / (earth_period/86400)) - (1 / (mars_period/86400))), format = 'jd')  
print(f'Synodic Period between Earth and Mars is {synodic_period_days} days')


"""
Using JPL data to get postion and velocity of earth (satellite) at departure and mars (target) at arrival)
"""

solar_system_ephemeris.set('de432s')

# # number of synodic periods before and after launch date
# n = 2
# # loop through different synodic periods to confim validity
# synodic_multiples = list(range(-n,n,1))

results = []
# for synodic_multiple in synodic_multiples:

#     departure_date = departure_date + (synodic_multiple * synodic_period_days)
tof_range = list(range(100, 400, 1))



for tof_days in tof_range:

    tof = TimeDelta(tof_days, format='jd')
    arrival_date = departure_date + tof
    # print(f'{arrival_date}\n')

    # position vector of earth and mars (initial and final satellite positions) wrt to soloar system barycenter
    r1_earth_bary, v1_earth_bary = get_body_barycentric_posvel( 'earth', departure_date)
    r2_mars_bary, v2_mars_bary = get_body_barycentric_posvel('mars', departure_date+tof)

    """
    Need to get sun position and velocity to transform
    earth and mars baycentric cords to helio-centric: sun centered inertial frame
    """

    # Position of Sun at departure and arrival
    r_sun1, v_sun1 = get_body_barycentric_posvel('sun', departure_date)
    r_sun2, v_sun2 = get_body_barycentric_posvel('sun', departure_date+tof)

    """
    heliocentric position and velocity vectors 
    """

    # Position & Velocity of earth (satellite) wrt respect to sun @ Depature
    r1_earth = (r1_earth_bary.xyz - r_sun1.xyz).to(u.km).value  # type:ignore
    v1_earth = (v1_earth_bary.xyz - v_sun1.xyz).to(u.km/u.s).value  # type:ignore

    # Position & Velocity of mars (satellite) wrtrespect to sun @ Depature
    r2_mars = (r2_mars_bary.xyz - r_sun2.xyz).to(u.km).value  # type:ignore
    v2_mars = (v2_mars_bary.xyz - v_sun2.xyz).to(u.km/u.s).value

    # cant get much frmo this since its restricted to 0 - pi
    transfer_angle = (np.arccos(np.dot(r1_earth, r2_mars) / (np.linalg.norm(r1_earth) * np.linalg.norm(r2_mars))))
    h = np.cross(r1_earth, r2_mars)
    if h[2] < 0:
        transfer_angle = 2 * np.pi - transfer_angle
        
    # print(f'Transfer angle: {np.degrees(transfer_angle):.1f}°')

    # goes back to what poliastro was talking about: prograde and retrograde

    # print(f------------------------------------------------------------------------------------------------For TOF of {tof_days} days------------------------------------------------------------------------------------------------n')
    # print(f'Earth Position at Depature: {r1_earth} km')
    # print(f'Mars Position at Arrival: {r2_mars} km\n')

    bodies =  [earth, venus, mercury, mars, jupiter, saturn, uranus, neptune]  
    central_body = sun
    fun_arg = [central_body, bodies]

    # print("VALLADO FUNCTION")
    # (v1_short, v2_short), = vallado.lambert(k, r1_earth*u.km, r2_mars*u.km, (tof.sec*u.s), short=True)
    # print(f"Departure velocity: {v1_short} | Arrival velocity: {v2_short}\n")

    # print("VRAJ FUNCTION") -> Universal Lambert Formulation
    transfer_short.a, transfer_short.e, transfer_v1_short, transfer_v2_short = universal_lambert( r1_earth, r2_mars, (tof.sec), transfer_short.mu, desired_path= 'short')
    # print(f'Short Transfer semi major axis is {transfer_short.a} km -->  {(transfer_short.a/149597870.7)} AU | Eccentricity = {transfer_short.e} | Departure velocity: {transfer_v1} km/s | Arrival velocity: {transfer_v2} km/s\n')
    C3_short = np.linalg.norm(transfer_v1_short - v1_earth)**2  
    transfer_long.a, transfer_long.e, transfer_v1_long, transfer_v2_long = universal_lambert( r1_earth, r2_mars, (tof.sec), transfer_long.mu, desired_path= 'long')
    # print(f'Long Transfer semi major axis is {transfer_long.a} km -->  {(transfer_long.a/149597870.7)} AU | Eccentricity = {transfer_long.e} | Departure velocity: {transfer_v1} km/s | Arrival velocity: {transfer_v2} km/s\n')
    C3_long = np.linalg.norm(transfer_v1_long - v1_earth)**2

    if C3_short < C3_long:
        C3 = C3_short   
        transfer_v1 = transfer_v1_short
        transfer_v2 = transfer_v2_short
    else:
        C3 = C3_long
        transfer_v1 = transfer_v1_long
        transfer_v2 = transfer_v2_long

    Vinf_arrival = (transfer_v2 - v2_mars) # arrival hyperbolic excess velcity in the MCI frame
    Vinf_departure = (transfer_v1 - v1_earth) # departure hyperbolic excess velociy in the ECI frame

    results.append({
            'tof_days': tof_days,
            'C3': C3,
            'V1': transfer_v1,
            'V2': transfer_v2,
            'V_earth': v1_earth,
            'V_mars': v2_mars,
            'V_inf_dep': Vinf_departure,
            'Vinf_arrival': Vinf_arrival,
            'r1': r1_earth,
            'r2': r2_mars,
            'arrival_date': arrival_date,
            'transfer_angle': np.degrees(transfer_angle)
     })
    # results.append({
    #     'synodic_multiple': synodic_multiple,
    #     'departure_date': departure_date,
    #     'tof_days': tof_days,
    #     'C3': C3,
    #     'V1': transfer_v1,
    #     'V2': transfer_v2,
    #     'V_earth': v1_earth,
    #     'V_mars': v2_mars,
    #     'V_inf_dep': np.sqrt(C3),
    #     'Vinf_arrival': Vinf_arrival,
    #     'r1': r1_earth,
    #     'r2': r2_mars,
    #     'arrival_date': arrival_date,
    #     'transfer_angle': np.degrees(transfer_angle)
    # })

# for synodic_multiple in synodic_multiples:
#     # Filter results for current synodic period
#     period_results = [r for r in results if r['synodic_multiple'] == synodic_multiple]
    
#     if period_results:
#         launch_date = period_results[0]['departure_date'].iso[:10]
        
#         print("\n" + "="*120)
#         print(f"LAUNCH DATE: {launch_date} (Synodic Period Offset: {synodic_multiple:+d})")
#         print("="*120)
#         print(f"{'TOF (days)':<12} {'C3 (km²/s²)':<15} {'V∞ Dep (km/s)':<16} {'V∞ Arr (km/s)':<16} {'Transfer Angle (°)':<20} {'Arrival Date':<20}")
#         print("="*120)
        
#         for res in period_results:
#             print(f"{res['tof_days']:<12} {res['C3']:<15.2f} {res['V_inf_dep']:<16.3f} "
#                   f"{res['Vinf_arrival']:<16.3f} {res['transfer_angle']:<20.5f} {res['arrival_date'].iso[:10]:<20}")
        
#         print("="*120)


"""
# Display results in a table
"""
# Syntax was taken from Claude im not gonna lie LOL

# Spike in the p

print("\n" + "="*140)
print(f"{'TOF (days)':<12} {'C3 (km²/s²)':<15} {'V∞ Dep (km/s)':<16} {'V∞ Arr (km/s)':<16} {'Transfer Angle (°)':<20} {'Arrival Date':<20}")
print("="*140)

for res in results:
    print(f"{res['tof_days']:<12} {res['C3']:<15.2f} {np.linalg.norm(res['V_inf_dep']):<16.3f} "
          f"{np.linalg.norm(res['Vinf_arrival']):<16.3f} {res['transfer_angle']:<20.5f} {res['arrival_date'].iso[:10]:<20}")

print("="*140 + "\n")

"""
Finding optimal solution based on minimum C3 & Vinf at arrival. 
"""

def find_optimal_solution(results, weight_C3, weight_Vinf): 

    # extracting C3 and Vinf arrival values
    C3_values = np.array([r['C3'] for r in results])
    Vinf_arrival_vectors = np.array([r['Vinf_arrival'] for r in results])
    Vinf_arrival_mag = np.linalg.norm(Vinf_arrival_vectors, axis=1)
    Vinf_departure_vectors = np.array([r['V_inf_dep'] for r in results])
    transfer_v1_vectors = np.array([r['V1'] for r in results])
    # r1_earth_vectors = np.array([r['r1'] for r in results])
    # v1_earth_vectors = np.array([r['V_earth'] for r in results])
    r2_mars_vectors = np.array([r['r2'] for r in results])
    v2_mars_vectors = np.array([r['V_mars'] for r in results])
    arrival_dates = np.array([r['arrival_date'] for r in results])
    
    # tof_values = np.array([r['tof_days'] for r in results])

    # normalzing them to be between 0 and 1 
    C3_norm = (C3_values - np.min(C3_values)) / (np.max(C3_values) - np.min(C3_values))
    Vinf_arrival_norm = (Vinf_arrival_mag - np.min(Vinf_arrival_mag)) / (np.max(Vinf_arrival_mag) - np.min(Vinf_arrival_mag))

    # give weighted scoe
    score = (weight_C3 * C3_norm) + (weight_Vinf * Vinf_arrival_norm)
    # minimize
    optimal_idx = np.argmin(score)
    optimal_C3 = C3_values[optimal_idx]
    optimal_Vinf_departure = Vinf_departure_vectors[optimal_idx]
    optimal_Vinf_arrival = Vinf_arrival_vectors[optimal_idx]
    optimal_transfer_v1 = transfer_v1_vectors[optimal_idx]
    # optimal_departure_earth_r1 = r1_earth_vectors[optimal_idx]
    # optimal_depature_earth_v1 = v1_earth_vectors[optimal_idx]
    optimal_arrival_mars_r2 = r2_mars_vectors[optimal_idx]
    optimal_arrival_mars_v2 = v2_mars_vectors[optimal_idx]
    arrival_date = arrival_dates[optimal_idx]
    print(f"\nOptimal Mission Duration: {results[optimal_idx]['tof_days']} Days. Arrival Date = {[arrival_date]} with (C3: {optimal_C3:.3f} km²/s², Vinf Arrival: {np.linalg.norm(optimal_Vinf_arrival):.3f} km/s, Vinf Departure: {np.linalg.norm(optimal_Vinf_departure):.3f} km/s)\n")
    return optimal_C3, optimal_Vinf_departure, optimal_Vinf_arrival, optimal_transfer_v1, optimal_arrival_mars_r2,optimal_arrival_mars_v2, arrival_date

# outputs array of optimal C3 & Vinf arrival based on assigned weights ( user defined )
optimal_C3, optimal_Vinf_departure, optimal_Vinf_arrival, optimal_transfer_v1,r2_mars,v2_mars, arrival_date = find_optimal_solution(results, weight_C3=0.75, weight_Vinf=0.25)
C3 = optimal_C3

Vinf_departure = optimal_Vinf_departure #*(u.km/u.s)
Vinf_arrival = optimal_Vinf_arrival #*(u.km/u.s)
transfer_v1 = optimal_transfer_v1

# V infinty arival and departure magnitudes 
Vinf_arrival_mag = np.linalg.norm(optimal_Vinf_arrival) #*(u.km/u.s)
Vinf_departure_mag = np.linalg.norm(optimal_Vinf_departure) #*(u.km/u.s)

# ----------------------------------------------------------------------------------------------N-body Propagation of Lamberts Problem-------------------------------------------------------------------------------------------------

# central_body = sun
# bodies = [mercury,venus,jupiter,saturn,uranus,neptune]
# fun_arg = [central_body,bodies]

# dt = TimeDelta(3600, format='sec')
# r_sats, _, _ = propagate_rk4(r1_earth, transfer_v1, departure_date, arrival_date, dt, fun_arg=fun_arg)

# r_mars_miss = r_sats[-1] - r2_mars
# print(f'Satellite Missed Mars Target by {np.linalg.norm(r_mars_miss):.5f} km')

# # # --> work on diff eq corrector

# # # thinking of doing newton raphson with f and g functions --> iterate v(_,_,_) = 0 --> proceed normally

# '''
# # TLDR --> Satellite misses mars by 107720.77453445194 KM!! pretty nice initial guess
# # propagate the new satellite initial conditions after adjusting RAAN and AOP
# '''

# -----------------------------------------------------------------------------------------------------------Parking Orbits------------------------------------------------------------------------------------------------------------

# Define Earth Parking Orbit
earth_parking = Orbit(mu=EARTH_MU,
                      a=(400+6378)*u.km,
                      e=0.0*u.km/u.km,  # unitless
                      f0=(0*u.deg).to(u.rad),
                      inc=(28.5*u.deg).to(u.rad),
                      raan=(175*u.deg).to(u.rad),
                      aop=(240*u.deg).to(u.rad)
                      )

# Define Mars Parking Orbit
mars_parking = Orbit(mu=MARS_MU,
                      a=(400+3396)*u.km,
                      e=0.0*u.km/u.km,  # unitless
                      f0=(0*u.deg).to(u.rad),
                      inc=(20*u.deg).to(u.rad),
                      raan=(55*u.deg).to(u.rad),
                      aop=(140*u.deg).to(u.rad)
                      )

# -----------------------------------------------------------------------------------------N-body Propagation of Parking Orbit post Delta V--------------------------------------------------------------------------------------------

# Numerically converged Transfer Velocity V1: [-15.49455038  26.46753395  11.89308889]

def convert_state_earth_to_helio(r_eci, v_eci,t):
    # Everything here is ICRS. 'eci' is just a label for 'Earth-Relative ICRS':
    r_earth_bary, v_earth_bary = get_body_barycentric_posvel('earth', t)
    r_sun_bary, v_sun_bary = get_body_barycentric_posvel('sun', t)
    #Vector from Sun to Earth in ICRS
    r_earth_helio = (r_earth_bary.xyz - r_sun_bary.xyz).to(u.km).value
    v_earth_helio = (v_earth_bary.xyz - v_sun_bary.xyz).to(u.km/u.s).value
    # Sat position wrt Sun = (Sun->Earth) + (Earth->Sat)
    r_helio = r_earth_helio + r_eci     # satellite pos wrt Sun, ecliptic
    v_helio = v_earth_helio + v_eci     # satellite vel wrt Sun, ecliptic
    return r_helio, v_helio

def convert_state_helio_to_mars(r_helio, v_helio,t):
    # Everything here is ICRS
    r_mars_bary, v_mars_bary = get_body_barycentric_posvel('mars', t)
    r_sun_bary, v_sun_bary = get_body_barycentric_posvel('sun', t)
    #Vector from Sun to Mars in ICRS: 
    r_mars_helio = (r_mars_bary.xyz - r_sun_bary.xyz).to(u.km).value
    v_mars_helio = (v_mars_bary.xyz - v_sun_bary.xyz).to(u.km/u.s).value
    # Sat position wrt Sun = (Sun->Mars) - (Sun->Mars) 
    r_sat_mars = r_helio - r_mars_helio
    v_sat_mars = v_helio - v_mars_helio
    return r_sat_mars , v_sat_mars

#leg based fraemwork --> resolve rk4 single step and calculate until it hits respecitve SOI's --. changes central bodies
def variable_nbody_propagtion(r0, v0, earth_soi, mars_soi, t0, tf):

    results = {'Leg 1 Earth Central': [], 'Leg 2 Heliocentric':[], 'Leg 3 Mars Central':[]}

    # ───────────────────────────────────────────────Phase 1: Earth-centered ───────────────────────────────────────────────
    print('Beginning Leg 1: Propagating until Earth SOI: Earth Centered')
    central_body = earth 
    perturbing = [mercury ,venus, moon, mars, jupiter, saturn, uranus,neptune, sun]
    fun_arg = [central_body,perturbing]

    r_eci , v_eci = r0.copy(), v0.copy()
    t = t0
    pos_vecs, vel_vecs, t_vals = [r_eci.copy()], [v_eci.copy()],[t]
    dt = TimeDelta(60, format='sec')

    while True:
      y = RK4_single_step(y_dot_n_ephemeris, dt, t, np.concatenate([r_eci,v_eci]), fun_arg)
      r_eci = y[:3]
      v_eci = y[3:6]
    
      t_curr = t
      t = t + dt

      pos_vecs.append(r_eci.copy())
      vel_vecs.append(v_eci.copy())
      t_vals.append(t_curr)

      if np.linalg.norm(r_eci) > earth_soi:
         print(f'Satellite Crossed Earth SOI at r = {np.linalg.norm(r_eci)} km from earth on {t_curr.iso}\n')
         t = t_curr # reset t to the time of crossing SOI for next leg
         break

    results['Leg 1 Earth Central'] = {'r':pos_vecs, 'v':vel_vecs, 't': t_vals}


        # ───────────────────────────────────────────────Phase 2: Sun-centered ───────────────────────────────────────────────
    print('Beginning Leg 2: Propagating until Mars SOI: Heliocentric')
    r,v = convert_state_earth_to_helio(r_eci,v_eci,t)
    central_body = sun 
    perturbing = [mercury ,venus, earth,moon, mars, jupiter, saturn, uranus,neptune]
    fun_arg = [central_body,perturbing]

    r_helio , v_helio = r.copy() , v.copy()
    print("Injected v:", v_helio)
    #Injected v: [-16.58054556  26.60949911  10.62554537]    
    pos_vecs, vel_vecs, t_vals = [r_helio.copy()], [v_helio.copy()],[t]
    dt = TimeDelta(3600, format='sec')
    mars_miss = []
    while True:
      y = RK4_single_step(y_dot_n_ephemeris, dt, t, np.concatenate([r_helio,v_helio]), fun_arg)
      r_helio = y[:3]
      v_helio = y[3:6]
      t_curr = t
      t = t + dt # for next iteration step
      pos_vecs.append(r_helio.copy())
      vel_vecs.append(v_helio.copy())
      t_vals.append(t_curr)

      r_mars_current,_ = get_body_barycentric_posvel('mars',t_curr)
      r_sun_current,_ = get_body_barycentric_posvel('sun',t_curr)
      r_mars_helio = (r_mars_current.xyz - r_sun_current.xyz).to(u.km).value # dist from sun to mars
      sat_mars_dist = np.linalg.norm(r_helio - r_mars_helio)
      mars_miss.append(sat_mars_dist)
      if sat_mars_dist < mars_soi:
        print(f'Satellite Crossed Mars SOI at r = {sat_mars_dist} km from Mars on {t_curr.iso}\n')
        t = t_curr
        break
      
      if t >= tf:
        current_miss = np.min(mars_miss)
        print(f"WARNING: reached tf. Closest approach to Mars: {current_miss:.0f} km "f"(Mars SOI = {mars_soi:.0f} km)\n")
        break
    results['Leg 2 Heliocentric'] = {'r':pos_vecs, 'v':vel_vecs, 't': t_vals}

    # ───────────────────────────────────────────────Phase 3: Mars-centered ───────────────────────────────────────────────
    print('Beginning Leg 3: Propagating until Arrival Date/time: Mars Centered')
    r,v = convert_state_helio_to_mars(r_helio,v_helio,t_curr)  
    central_body = mars
    perturbing = [mercury ,venus, earth, moon, jupiter, saturn, uranus,neptune, sun]
    fun_arg = [central_body,perturbing]

    t = t_curr
    r_mci , v_mci = r.copy() , v.copy()
    
    pos_vecs, vel_vecs, t_vals = [r_mci.copy()], [v_mci.copy()],[t]
    dt = TimeDelta(60, format='sec')
    r_dot_prev = np.dot(r_mci, v_mci) / np.linalg.norm(r_mci)  # radial velocity
    periapsis_state = None

    while t < (tf + TimeDelta(10, format='jd')):
       y = RK4_single_step(y_dot_n_ephemeris ,dt ,t ,np.concatenate([r_mci,v_mci]),fun_arg)
       r_mci = y[:3]
       v_mci = y[3:6]
       t_curr = t
       t = t+dt

       # we know that at mars periapsis: 
          # r_mci should be the closest to mars surface i.e the minimum r value
          # at periapsis, the velocity and position are perpendicular --> r dot v = 0 

       r_dot_curr = np.dot(r_mci,v_mci)/np.linalg.norm(r_mci)

       if r_dot_prev < 0 and r_dot_curr >= 0: 
            print(f"Periapsis detected at t = {t_curr.iso} with r = {np.linalg.norm(r_mci):.3f} km and v = {np.linalg.norm(v_mci):.3f} km/s \n")
            periapsis_state = (r_mci.copy(), v_mci.copy(), t_curr)
            break
    
       r_dot_prev = r_dot_curr

       pos_vecs.append(r_mci.copy())
       vel_vecs.append(v_mci.copy())
       t_vals.append(t_curr)
    print(f" At end of propagation: t = {t_curr.iso}, Distance from Mars = {np.linalg.norm(r_mci):.5f} km, velocity wrt Mars = {np.linalg.norm(v_mci):.5f} km/s")
    results['Leg 3 Mars Central'] = {'r':pos_vecs, 'v':vel_vecs, 't': t_vals, 'periapsis': periapsis_state}
    
    return results
# ---------------------------------------------------------------------------------------------------=----B-plane Targetting-------=---------------------------------------------------------------------------------------------------

# propagate with new initial conditions from parking orbit targeting
def sphere_of_influence(body, sun_mu):
  
    a = body.a.value  # semi-major axis of planet around sun (km)
    body.mu = G.value * body.mass.value  
    mu_ratio = body.mu / sun_mu
    r_soi = a * (mu_ratio)**(2/5)
    return r_soi

earth_soi = sphere_of_influence(earth, SUN_MU)
mars_soi = sphere_of_influence(mars, SUN_MU)

def Bplane2(r_soi_cross,vinf_arrival_vec,mars_mu):

    vinf_arrival = np.linalg.norm(vinf_arrival_vec)

    # all the vectors are in the perifocal frame 
    h = np.cross(r_soi_cross, vinf_arrival_vec)
    h_hat = h / np.linalg.norm(h)    
    
    e_vec = (1/mars_mu) * (vinf_arrival**2 * r_soi_cross - np.dot(r_soi_cross,vinf_arrival_vec)*vinf_arrival_vec) - r_soi_cross/np.linalg.norm(r_soi_cross)
    e = np.linalg.norm(e_vec)       

    a = -mars_mu/vinf_arrival**2
    P_mag = mars_mu * e # same as -vinf**2 * a *e
    P_hat = e_vec/e
    P = P_mag * P_hat

    Q = np.cross(h,P)
    Q_hat = Q/np.linalg.norm(Q)

    cos_finf = -mars_mu/P_mag
    sin_finf = -1*np.sqrt(1-(mars_mu/P_mag)**2)

    s_hat = -(cos_finf * P_hat + sin_finf * Q_hat)
    # print(f's_hat = {s_hat}')
    N = np.array([0,0,1])

    '''
     N = np.array([0,0,1]) Because you are in ICRS, this  vector points toward the Earth's North Pole.
     While this is mathematically valid for defining a coordinate system,
             most Mars mission planners define the B-plane relative to the Mars Orbital Plane or the Mars North Pole. 
             Using the Earth's pole is fine for convergence, but your btheta value will be relative to Earth's equator, not the Martian landscape. 
             Just something to keep in mind when interpreting your results!
    
    '''

    t_hat = np.cross(s_hat,N)/np.linalg.norm(np.cross(s_hat,N))
    r_hat = np.cross(s_hat,t_hat)

    B_vec= (1/vinf_arrival) * np.cross(s_hat,h)
    B = np.linalg.norm(B_vec)
    # print(f'B vector from UC boulder {B_vec}')

    rp = -mars_mu/vinf_arrival**2 + np.sqrt((mars_mu/vinf_arrival**2)**2 + B**2)
    print(f' the close approach distance is: {rp}\n')

    B_theta = np.arccos( np.dot(B_vec,t_hat)/B )
    BR = np.dot(B_vec,r_hat)
    BT = np.dot(B_vec,t_hat)
    
    return rp,B_theta,BR,BT

# ----------------------------------------------------------------------------------------Analytically Calculating the Earth-Vinfinity vector ------------------------------------------------------------------------------------------

def calculate_vinf_departure(dV,orbit,hyp_parameter):

    # parking orbit state
    r_pqw, v_pqw = orb_2_pqw(orbit.r_at_true_anomaly(orbit.f0).value,orbit.f0.value, orbit.e.value,orbit.p.value, orbit.mu.value)
    r_eci, v_eci = perif_2_eci(r_pqw, v_pqw, orbit.inc, orbit.raan, orbit.aop)
    rp = orbit.a.value * (1-orbit.e.value)
    vinf_deptarture = np.sqrt( (np.linalg.norm(v_eci) + dV)**2 - (2*orbit.mu.value/rp))

    a_hyp = -orbit.mu.value/vinf_deptarture**2
    e_hyp = 1 - (rp/a_hyp)

    # v_PQW(f) = sqrt(mu/p) * [-sin(f), e + cos(f), 0]
    # Substituting f = f_inf  (cos f = -1/e, sin f = sqrt(e^2-1)/e
    vinf_pqw = vinf_deptarture * np.array([-1/e_hyp , np.sqrt(e_hyp**2-1)/e_hyp,0])

    # raan, aop, and inc of hyperbolic orbit
    inc_hyp = hyp_parameter[3]
    raan_hyp = hyp_parameter[4]
    aop_hyp = hyp_parameter[5]

    _,vinf_eci = perif_2_eci(np.array([0,0,0]),vinf_pqw, inc_hyp, raan_hyp, aop_hyp)

    return vinf_eci


def hyperbolic_parameters(rp,v_postburn,orbit): 

    h_vec = np.cross(rp,v_postburn)
    h = np.linalg.norm(h_vec)
    P = h**2/orbit.mu.value

    e_vec = 1/orbit.mu.value * (np.linalg.norm(v_postburn)**2 * rp - np.dot(rp,v_postburn)*v_postburn) - rp/np.linalg.norm(rp)
    e = np.linalg.norm(e_vec)
    a = P/(e**2 -1)
    
    Node = np.array([0,0,1])
    
    inc = np.arccos(h_vec[2]/np.linalg.norm(h_vec))
    N = np.cross(Node,h_vec)

    if N[1] >= 0: 
        raan  = np.arccos(N[0]/np.linalg.norm(N))
    else: 
        raan = 2*np.pi - np.arccos(N[0]/np.linalg.norm(N))

    if e_vec[2] >= 0:
        aop = np.arccos(np.dot(N,e_vec)/(np.linalg.norm(N)*np.linalg.norm(e_vec)))
    else:
        aop = 2*np.pi - np.arccos(np.dot(N,e_vec)/(np.linalg.norm(N)*np.linalg.norm(e_vec)))

    hyp_parameter = np.array([a,e,P,inc, raan, aop])

    return hyp_parameter

def orbit_to_inertial_state(orbit):
    r_pqw, v_pqw = orb_2_pqw(orbit.r_at_true_anomaly(orbit.f0).value,orbit.f0.value, orbit.e.value,orbit.p.value, orbit.mu.value)
    r_eci, v_eci = perif_2_eci(r_pqw, v_pqw, orbit.inc, orbit.raan, orbit.aop)
    return r_eci,v_eci

# r_eci, v_eci = orbit_to_inertial_state(earth_parking)
# dV0 = np.sqrt(Vinf_departure_mag**2 + (2*EARTH_MU.value/np.linalg.norm(r_eci))) - np.linalg.norm(v_eci)
# v_postburn_eci = v_eci + (dV0 * (v_eci/np.linalg.norm(v_eci)))  # apply prograde delta 
# hyp_parameters = hyperbolic_parameters(r_eci, v_postburn_eci, earth_parking)
# vinf_eci = calculate_vinf_departure(dV0, earth_parking, hyp_parameters)
# print(f'Calculated Vinf departure vector from analytical function: {vinf_eci} km')

# ------------------------------------------------------------------------Step 3: Differential Correction: targetting Lamberts Vinf for better initial conditions ----------------------------------------------------------------------
print("\n------------------------------------------------------------------------------------------------Phase 1: Vinf Targetting------------------------------------------------------------------------------------------------n")

# Guess X → propagate → detect periapsis → compute B-plane → correct X
def vinf_target_function(x,fun_args):

    orbit = fun_args[0]
    orbit.raan = x[0][0]
    orbit.aop  = x[1][0]
    dV         = x[2][0]

    r_eci, v_eci = orbit_to_inertial_state(orbit)
    v_postburn_eci = v_eci + (dV * (v_eci/np.linalg.norm(v_eci)))  # apply prograde delta V
    hyp_parameters = hyperbolic_parameters(r_eci, v_postburn_eci, orbit)
    vinf_eci = calculate_vinf_departure(dV, orbit, hyp_parameters)
    return vinf_eci.reshape(3,1)


# Forward differences
def sensitivity_matrix(x, target_function, fun_args, step_sizes, f_x):
    orbit = fun_args[0]
    # Reshape dt_input args into dt vectors
    dt_rann_ar = np.array([step_sizes[0][0], 0,0]).reshape(3, 1)
    dt_aop_ar = np.array([0, step_sizes[1][0],0]).reshape(3, 1)
    dt_dV_ar = np.array([0, 0, step_sizes[2][0]]).reshape(3, 1)
    # equations from AGI newtons method paper 
    dt_raan_col = (1/(step_sizes[0][0]))*(target_function( x + dt_rann_ar,fun_args) - f_x)
    # Reset orbit to x before each call so state doesn't bleed between columns
    orbit.raan = x[0][0]  # reset
    orbit.aop  = x[1][0]  # reset
    dt_aop_col = (1/(step_sizes[1][0]))*(target_function( x + dt_aop_ar,fun_args) - f_x)
    orbit.raan = x[0][0]  # reset
    orbit.aop  = x[1][0]  # reset
    dt_dV_col = (1/(step_sizes[2][0]))*(target_function(x + dt_dV_ar,fun_args) - f_x)
    orbit.raan = x[0][0]  # reset
    orbit.aop  = x[1][0]  # reset
    return np.block([dt_raan_col, dt_aop_col, dt_dV_col])

## central difference   
# def sensitivity_matrix(x, target_function, fun_args, step_sizes, f_x):
#     orbit = fun_args[0]
#     # Reshape dt_input args into dt vectors
#     dt_rann_ar = np.array([step_sizes[0][0], 0,0]).reshape(3, 1)
#     dt_aop_ar = np.array([0, step_sizes[1][0],0]).reshape(3, 1)
#     dt_dV_ar = np.array([0, 0, step_sizes[2][0]]).reshape(3, 1)
#     # equations from AGI newtons method paper 
#     dt_raan_col = (1/( 2 * step_sizes[0][0]))*(target_function( x + dt_rann_ar,fun_args) - target_function( x - dt_rann_ar,fun_args))
#     # Reset orbit to x before each call so state doesn't bleed between columns
#     orbit.raan = x[0][0]  # reset
#     orbit.aop  = x[1][0]  # reset
#     dt_aop_col = (1/(2 * step_sizes[1][0]))*(target_function( x + dt_aop_ar,fun_args) - target_function( x - dt_aop_ar,fun_args))
#     orbit.raan = x[0][0]  # reset
#     orbit.aop  = x[1][0]  # reset
#     dt_dV_col = (1/(2 *step_sizes[2][0]))*(target_function(x + dt_dV_ar,fun_args) - target_function(x - dt_dV_ar,fun_args))
#     orbit.raan = x[0][0]  # reset
#     orbit.aop  = x[1][0]  # reset
#     return np.block([dt_raan_col, dt_aop_col, dt_dV_col])

def differential_correction(
        x0,
        y_d,
        targetting_function, 
        function_args = None,
        step_sizes = np.array([np.deg2rad(.01), np.deg2rad(.01), .01]).reshape(3, 1),
        tol = np.array([10e-4, 10e-4, 10e-4]).reshape(3, 1),
        max_i = 50,
        orbit = None,
):

    x = x0.copy()
    for i in range(max_i):
        orbit.raan = x[0][0]
        orbit.aop  = x[1][0]

        f_x = targetting_function(x, function_args)
        error = (f_x-y_d)
        J = sensitivity_matrix(x,targetting_function, function_args, step_sizes, f_x)    
        x_k = x - np.linalg.pinv(J)@(f_x-y_d)

        if targetting_function == vinf_target_function:
            print(f"[{i}] Difference between computed and target Vinf: {error.flatten()} km/s --> {np.linalg.norm(error.flatten()):.4f} km/s")
            print(f"Parking Orbit RAAN = {np.rad2deg(x[0][0])} deg  | Parking Orbit AOP = {np.rad2deg(x[1][0])} deg | dV = {x[2][0]} km/s\n")
        elif targetting_function == mars_position_target_function:
            print(f"[{i}] Difference between computed and target Mars COM position: {error.flatten()} km --> {np.linalg.norm(error.flatten()):.4f} km")
            print(f"Parking Orbit RAAN = {np.rad2deg(x[0][0])} deg  | Parking Orbit AOP = {np.rad2deg(x[1][0])} deg | dV = {x[2][0]} km/s\n")
        elif targetting_function == bplane_target_function:
            print(f"[{i}] Difference between computed and target B-plane parameters: {error.flatten()}. Closest approach = {f_x.flatten()[0]} km vs target of {y_d.flatten()[0]} km")
            print(f"Parking Orbit RAAN = {np.rad2deg(x[0][0])} deg  | Parking Orbit AOP = {np.rad2deg(x[1][0])} deg | dV = {x[2][0]} km/s\n")

        x = x_k
        i += 1

        if np.all(np.abs(error) < tol):
            print(f"[CONVERGED] ERROR:{error.flatten()}")
            break

        if i > max_i:
            print(f"[MAX ITER] ERROR:{error.flatten()}")
            break

    orbit.raan = x[0][0]  # reset
    orbit.aop  = x[1][0]  # reset
    dV = x[2][0]

    f_x = targetting_function(x, function_args)
    error = (f_x-y_d)
    return x,f_x,error

x ,f_x, error = differential_correction(
    x0 = np.array([np.deg2rad(175.0), np.deg2rad(240), 3.55]).reshape(3, 1),
    y_d = Vinf_departure.reshape(3,1),
    targetting_function = vinf_target_function,
    function_args = (earth_parking,),
    step_sizes = np.array([np.deg2rad(.01), np.deg2rad(.01), .01]).reshape(3, 1),
    tol = np.array([10e-4, 10e-4, 10e-4]).reshape(3, 1),
    max_i = 50,
    orbit = earth_parking
)
# Parking Orbit RAAN = 91.09956973620143 deg  | Parking Orbit AOP = 265.05943748625856 deg | dV = 3.637 km/s
dV = x[2][0]
print(f"\nParking Orbit RAAN = {np.rad2deg(x[0][0])} deg  | Parking Orbit AOP = {np.rad2deg(x[1][0])} deg | dV = {x[2][0]} km/s\n")
print(f'Vinf departure after targeting: {f_x.flatten()} km/s with error of {error.flatten()} km/s compared to target Vinf departure of {Vinf_departure.flatten()} km/s\n')

# --------------------------------------------------------------------------- The zero-SOI framing/Two-body continuation first --> Purely heliocentric motions -------------------------------------------------------------------------

# r_eci, v_eci = orbit_to_inertial_state(earth_parking)
# v_postburn_eci = v_eci + dV * v_eci/np.linalg.norm(v_eci)  

# central_body = sun
# bodies = []
# fun_arg = [central_body,bodies]

# dt = TimeDelta(3600, format='sec')
# r_sats, _, _ = propagate_rk4(r_eci+r1_earth, f_x.reshape(1,3)+ v1_earth, departure_date, arrival_date, dt, fun_arg=fun_arg)

# miss_vector = []
# miss_mag = []

# for pos in r_sats: 
#     r_mars_miss = pos - r2_mars
#     miss_vector.append(r_mars_miss)
#     r_miss_mag = np.linalg.norm(r_mars_miss)
#     miss_mag.append(r_miss_mag)

# closest_idx =  np.argmin(miss_mag)
# r_mars_miss = miss_vector[closest_idx]
# print(f'Satellite Missed Mars Target by {np.linalg.norm(r_mars_miss):.5f} km')
# # np.float64(76003.59582051697) when dt = 60. dont run. takes 2.5 hrs

'''
when i run the full nbody propagation here with the 3 leg mission, im able to cross earth & mars SOI, and my mars periapsis is 256,000 km with a mars relative velocity of 2.645 km/s.
Nbody_prop = variable_nbody_propagtion(r_eci, v_postburn_eci, earth_soi, mars_soi, departure_date, arrival_date)
'''

# ------------------------------------------------------------------------------Step 4: Differential Correction: targetting targetting: r_sc - r_mars = 0 ----------------------------------------------------------------------------

# want to target r_sc - r_mars = 0 at arrival date (or peripapsis event crossing) to get better initial conditions for departure state. This kills reliances on lambert all together. 
# we have kinda good initial conditions from lamberts. work on iterating on the inital RAAN/AOP/dV from parking orbit to miniminze the error between sat and mars. 
# use those converged conditions for RAAN/AOP/dV as initial conditions for the n-body propagator and then apply differential correciton once again for b plane correction 

print("\n------------------------------------------------------------------------------------Phase 2: R_mars_COM targetting: r_sc - r_mars = 0 ------------------------------------------------------------------------------------n")

def mars_position_target_function(x,fun_args):
    orbit,departure_date,arrival_date = fun_args
    orbit.raan = x[0][0]
    orbit.aop  = x[1][0]
    dV         = x[2][0]

    r_eci, v_eci = orbit_to_inertial_state(orbit)
    v_postburn_eci = v_eci + (dV * (v_eci/np.linalg.norm(v_eci)))  # apply prograde delta V
    hyp_parameters = hyperbolic_parameters(r_eci, v_postburn_eci, orbit)
    vinf_eci = calculate_vinf_departure(dV, orbit, hyp_parameters)

    dt = TimeDelta(3600, format='sec')
    r_sats, _, _ = propagate_rk4(r_eci+r1_earth, vinf_eci.reshape(1,3)+ v1_earth, departure_date, arrival_date, dt, fun_arg=[sun,[]])
    return r_sats[-1].reshape(3,1)

# actual code for running

x , f_x, error = differential_correction(
    x0 = np.array([x[0][0], x[1][0], x[2][0]]).reshape(3, 1),
    y_d = r2_mars.reshape(3,1),
    targetting_function = mars_position_target_function,
    function_args = (earth_parking, departure_date, arrival_date),
    step_sizes = np.array([np.deg2rad(.01), np.deg2rad(.01), .01]).reshape(3, 1),
    tol = np.array([10e-4, 10e-4, 10e-4]).reshape(3, 1),
    max_i = 50,
    orbit = earth_parking
)

# # temp code using the alr converged values from running true differntial correciton to same time
# x , f_x, error = differential_correction(
#     x0 = np.array([np.deg2rad(91.12437422103665), np.deg2rad(265.0703882273781), 3.6368109087080462]).reshape(3, 1),
#     y_d = r2_mars.reshape(3,1),
#     targetting_function = mars_position_target_function,
#     function_args = (earth_parking, departure_date, arrival_date),
#     step_sizes = np.array([np.deg2rad(.01), np.deg2rad(.01), .01]).reshape(3, 1),
#     tol = np.array([10e-4, 10e-4, 10e-4]).reshape(3, 1),
#     max_i = 50,
#     orbit = earth_parking
# )

print(f"\nParking Orbit RAAN = {np.rad2deg(x[0][0])} deg  | Parking Orbit AOP = {np.rad2deg(x[1][0])} deg | dV = {x[2][0]} km/s\n")
print(f'Heliocentric Position of Spacecraft at Mars arrival TOF: {f_x.flatten()} km with error of {error.flatten()} km compared to the target position {r2_mars.flatten()} km \n')
dV = x[2][0]

'''
iteration 6
[CONVERGED] ERROR:[-7.16149807e-05 -7.35819340e-05 -3.23206186e-05]
Parking Orbit RAAN = 91.12437422103665 deg  | Parking Orbit AOP = 265.0703882273781 deg | dV = 3.6368109087080462 km/s'''

# --------------------------------------------------------------------------------------Step 5: Differential Correction: Mars B-Plane Targettings---------------------------------------------------------------------------------------
print("\n------------------------------------------------------------------------------------------------Phase 3: B-Plane Targetting ------------------------------------------------------------------------------------------------n")
# earth_parking.raan = np.deg2rad(91.12437422103665)
# earth_parking.aop = np.deg2rad(265.0703882273781)
# dV = 3.6368109087080462
# r_eci, v_eci = orbit_to_inertial_state(earth_parking)
# v_postburn_eci = v_eci + (dV * (v_eci/np.linalg.norm(v_eci)))
# Nbody_prop = variable_nbody_propagtion(r_eci, v_postburn_eci, earth_soi, mars_soi, departure_date, arrival_date)

# print("hello")

def bplane_target_function(x, fun_args):
    orbit,earth_soi, mars_soi, departure_date, arrival_date = fun_args
    orbit.raan = x[0][0]
    orbit.aop  = x[1][0]
    dV = x[2][0]

    r_eci, v_eci = orbit_to_inertial_state(orbit)
    v_postburn_eci = v_eci + (dV * (v_eci/np.linalg.norm(v_eci)))
    Nbody_prop = variable_nbody_propagtion(r_eci, v_postburn_eci, earth_soi, mars_soi, departure_date, arrival_date)
    r_mars_soi_nbody = Nbody_prop['Leg 3 Mars Central']['r'][0]
    vinf_arrival_nbody = Nbody_prop['Leg 3 Mars Central']['v'][0]
    rp,B_theta,BR,BT = Bplane2(r_mars_soi_nbody,vinf_arrival_nbody,MARS_MU.value)
    return np.array([rp, B_theta]).reshape(2, 1)

x, f_x, error = differential_correction(
    x0 = np.array([x[0][0], x[1][0], x[2][0]]).reshape(3, 1),
    y_d = np.array([400+3396, np.deg2rad(0)]).reshape(2, 1),  # target rp of 400 km and Btheta of 0 deg
    targetting_function = bplane_target_function,
    function_args = (earth_parking, earth_soi, mars_soi, departure_date, arrival_date),
    step_sizes = np.array([np.deg2rad(.01), np.deg2rad(.01), .01]).reshape(3, 1),
    tol = np.array([10, np.deg2rad(0.01)]).reshape(2, 1),
    max_i = 50,
    orbit = earth_parking
)


'''
plan for b plane targetting:
    - some notes for myself: the B plane is a plane orthogonal/normal to the hyperbolic trajectory plane ( the incoming asymptote) and the initial hyperbolic excess velocity vector. 
        - normal to the vinf vector 
    - it allows s/c to have some specific hyperbolic trajectory for flyby or in our case: orbit capture
        B vector vector from target body COM to where the vinf hits b plane

     -B plane defined by 3 unit vectors: S, T, R and the B vector which lies in the plane defined by T & R, and the vertex angle between the B vector and the T vector 
    - S = unit vecotr in the direction of the hyperbolic excess velocity vector --> Vinf_arrival at the entry of SOI
        - S_hat is in perifocal cords of the planet
            - S_hat * P = -Pcos(f_inf) where cos(f_inf) = -mu/P
                - P = vector constant of integration in the direction of the hyperbola vertex (periapsis in perifocal frame --> like eccentricity): abs(P) = -Vinf^2 *ae

    - T = unit vector orthogonal to the S_hat and the normal of the planet. typically on the ecliptic plane of the solar system
    - R = unit vector: cross product S x T

    - B vector points from center of planet to the point at which the incoming asymptote of a spacecrafts hyperbolic trajectory pierces the B plane

Need to set up Mars Parking orbit 
--> need to iterate kinda like we did here where we need to define a position we want to be at --> iterate on either a dv manuever and or parking orbit to satisfy 

2  approaches: 

    1. iterate on the transfer V1 from departure: Perturb transfer_v1, propagate n-body to Mars SOI, measure B-plane error, iterate with Newton's method.
    2. Apply a small dV somewhere along the transfer, iterate on that [dvx, dvy, dvz] to minimize B-plane error.

    For both, need to extract the point wherein the n-body propagtor outputs the position & velocity of the s/c when entering MARS SOi
'''