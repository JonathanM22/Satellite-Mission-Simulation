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

# UNCOMMENT FOR FULL SWEEP
# loop through different synodic periods to confim validity
# n = 2 # number of synodic periods before and after launch date
# synodic_multiples = list(range(-n,n,1))
# start_departure_date = departure_date

results = []
# for synodic_multiple in synodic_multiples:

#     departure_date = start_departure_date + (synodic_multiple * synodic_period_days)

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
#             print(f"{res['tof_days']:<12} {res['C3']:<15.2f} {np.linalg.norm(res['V_inf_dep']):<16.3f} "
#                  f"{np.linalg.norm(res['Vinf_arrival']):<16.3f} {res['transfer_angle']:<20.5f} {res['arrival_date'].iso[:10]:<20}")
#         print("="*120)


"""
# Display results in a table
"""
# Syntax was taken from Claude im not gonna lie LOL

print("\n" + "="*140)
print(f"{'TOF (days)':<12} {'C3 (km²/s²)':<15} {'V∞ Dep (km/s)':<16} {'V∞ Arr (km/s)':<16} {'Transfer Angle (°)':<20} {'Arrival Date':<20}")
print("="*140)

for res in results:
    print(f"{res['tof_days']:<12} {res['C3']:<15.2f} {np.linalg.norm(res['V_inf_dep']):<16.6f} "
          f"{np.linalg.norm(res['Vinf_arrival']):<16.6f} {res['transfer_angle']:<20.5f} {res['arrival_date'].iso[:10]:<20}")

print("="*140 + "\n") 

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
    tof_days = np.array([r['tof_days'] for r in results])
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
    TOF = tof_days[optimal_idx]
    print(f"\nOptimal Mission Duration: TOF = {TOF} Days. Arrival Date = {[arrival_date]} with (C3: {optimal_C3:.3f} km²/s², Vinf Arrival: {np.linalg.norm(optimal_Vinf_arrival):.3f} km/s, Vinf Departure: {np.linalg.norm(optimal_Vinf_departure):.3f} km/s)\n")
    return optimal_C3, optimal_Vinf_departure, optimal_Vinf_arrival, optimal_transfer_v1, optimal_arrival_mars_r2,optimal_arrival_mars_v2, arrival_date, TOF

# outputs array of optimal C3 & Vinf arrival based on assigned weights ( user defined )
optimal_C3, optimal_Vinf_departure, optimal_Vinf_arrival, optimal_transfer_v1,r2_mars,v2_mars, arrival_date,TOF = find_optimal_solution(results, weight_C3=0.75, weight_Vinf=0.25)
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
                      e=0.0001*u.km/u.km,  # unitless
                      f0=(0*u.deg).to(u.rad),
                      inc=(28.5*u.deg).to(u.rad),
                      raan=(175*u.deg).to(u.rad),
                      aop=(240*u.deg).to(u.rad)
                      )

# Define Mars Parking Orbit
mars_parking = Orbit(mu=MARS_MU,
                      a=(400+3396)*u.km,
                      e=0.0001*u.km/u.km,  # unitless
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

def Bplane2(r_periapsis,v_periapsis,mars_mu):
    v_mag = np.linalg.norm(v_periapsis)
    r_mag = np.linalg.norm(r_periapsis)
    energy_term = .5*v_mag**2 - mars_mu/r_mag

    if energy_term <= 0:
            # trial trajectory is bound (elliptical) at "periapsis" -- not a valid
            # hyperbolic flyby state, B-plane math doesn't apply
            raise ValueError(f"Non-hyperbolic periapsis state (specific energy={energy_term/2:.6f} km²/s² <= 0); "
                            f"cannot compute B-plane parameters")

    vinf_arrival = np.sqrt(2*energy_term)
    h = np.cross(r_periapsis, v_periapsis)
    h_hat = h / np.linalg.norm(h)    
    
    e_vec = (1/mars_mu) * (v_mag**2 * r_periapsis - np.dot(r_periapsis,v_periapsis)*v_periapsis) - r_periapsis/np.linalg.norm(r_periapsis)
    e = np.linalg.norm(e_vec)       
    if e <= 1.0:
        raise ValueError(f"Non-hyperbolic eccentricity e={e:.6f} <= 1; cannot compute B-plane parameters")
    a = -mars_mu/vinf_arrival**2
    P_mag = mars_mu * e # same as -vinf**2 * a *e
    P_hat = e_vec/e
    P = P_mag * P_hat
    Q = np.cross(h,P)
    Q_hat = Q/np.linalg.norm(Q)

    cos_finf = -mars_mu/P_mag
    sin_finf = -1*np.sqrt(1-(mars_mu/P_mag)**2)
    
    s_hat = -(cos_finf * P_hat + sin_finf * Q_hat)
    N = np.array([0,0,1])  # ICRS frame. Here N can be anything
    # The B-plane T/R axes are a reporting convention, not a physical property of Mars.
    '''
    The fact that periapsis/SOI-entry is a Mars-centered event doesn't change what frame you report B-plane targets in.
    it's still ICRF-Z-referenced by convention, same as if you were characterizing a flyby of any other body. 
    It's only when you go from "where does the asymptote pierce the B-plane" to "what orbit do I actually fly around Mars" that Mars' physical pole enters the picture, which is exactly the solve_achievable_plane fix from before.
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
    """ Calculate the numerical Jacobian using forward finite differences.
    Columns:
    J[:,0] = dF/d(RAAN)
    J[:,1] = dF/d(AOP)
    J[:,2] = dF/d(dV) """
    orbit = fun_args[0]
    # Perturbation vectors
    dt_rann_ar = np.array([step_sizes[0][0], 0,0]).reshape(3, 1)
    dt_aop_ar = np.array([0, step_sizes[1][0],0]).reshape(3, 1)
    dt_dV_ar = np.array([0, 0, step_sizes[2][0]]).reshape(3, 1)
    if target_function.__name__ == "bplane_target_function":
        print(f"Perturbing Earth Parking orbit RAAN run:")
    # equations from AGI newtons method paper 
    dt_raan_col = (1/(step_sizes[0][0]))*(target_function( x + dt_rann_ar,fun_args) - f_x)
    # Reset orbit to x before each call so state doesn't bleed between columns
    orbit.raan = x[0][0]  # reset
    orbit.aop  = x[1][0]  # reset
    if target_function.__name__ == "bplane_target_function":
        print(f"Perturning Earth parking orbit AOP run: ")
    dt_aop_col = (1/(step_sizes[1][0]))*(target_function( x + dt_aop_ar,fun_args) - f_x)
    orbit.raan = x[0][0]  # reset
    orbit.aop  = x[1][0]  # reset
    if target_function.__name__ == "bplane_target_function":
        print(f"Perturbing dV at Earth burn run:")
    dt_dV_col = (1/(step_sizes[2][0]))*(target_function(x + dt_dV_ar,fun_args) - f_x)
    orbit.raan = x[0][0]  # reset
    orbit.aop  = x[1][0]  # reset
    return np.block([dt_raan_col, dt_aop_col, dt_dV_col])

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
        y_d_current = getattr(targetting_function, '_last_target', y_d)        
        error = (f_x-y_d_current)

        # snapshot nominal periapsis state before Jacobian perturbations overwrite it
        nominal_periapsis = getattr(targetting_function, '_last_periapsis', None)

        J = sensitivity_matrix(x,targetting_function, function_args, step_sizes, f_x) 

        # --- DIAGNOSTICS: paste here ---
        singular_values = np.linalg.svd(J, compute_uv=False)
        cond_number = np.linalg.cond(J)
        print(f"    J singular values: {singular_values} | cond(J) = {cond_number:.3e}")

        if targetting_function == vinf_target_function:
            print(f"[{i}] Difference between computed and target Vinf: {error.flatten()} km/s --> {np.linalg.norm(error.flatten()):.6f} km/s")
            print(f"Parking Orbit RAAN = {np.rad2deg(x[0][0])} deg  | Parking Orbit AOP = {np.rad2deg(x[1][0])} deg | dV = {x[2][0]} km/s\n")
        elif targetting_function == mars_position_target_function:
            print(f"[{i}] Difference between computed and target Mars COM position: {error.flatten()} km --> {np.linalg.norm(error.flatten()):.6f} km")
            print(f"Parking Orbit RAAN = {np.rad2deg(x[0][0])} deg  | Parking Orbit AOP = {np.rad2deg(x[1][0])} deg | dV = {x[2][0]} km/s\n")
        elif targetting_function == bplane_target_function:
            tof_diag = getattr(targetting_function, '_last_TOF', None)
            tof_str = f"{tof_diag:.3f} days" if tof_diag is not None else "N/A"
            print(f"[{i}] BR error: {error[0,0]:.6f} km | BT error: {error[1,0]:.6f} km | TOF (floating): {tof_str}")
            print(f"    Computed: BR={f_x[0,0]:.6f} km, BT={f_x[1,0]:.6f} km")
            print(f"    Target:   BR={y_d_current[0,0]:.6f} km, BT={y_d_current[1,0]:.6f} km")
            print(f"    RAAN={np.rad2deg(x[0][0]):.10f} deg | AOP={np.rad2deg(x[1][0]):.10f} deg | dV={x[2][0]:.10f} km/s")
            print(f"    Mars Parking Orbit --> inc={np.rad2deg(mars_parking.inc):.10f} deg | raan={np.rad2deg(mars_parking.raan):.10f} deg\n")
            if nominal_periapsis is not None:
                r_p_nom, v_p_nom, t_p_nom = nominal_periapsis
                print(f"    Mars miss distance at periapsis: {np.linalg.norm(r_p_nom):.3f} km | "
                    f"Velocity at periapsis: {np.linalg.norm(v_p_nom):.6f} km/s\n")


        if np.all(np.abs(error) < tol):
                    print(f"[CONVERGED] ERROR:{error.flatten()}")
                    break

        x_k = x - np.linalg.pinv(J)@error
        x = x_k
        i += 1

        if i > max_i:
            print(f"[MAX ITER] ERROR:{error.flatten()}")
            break

    orbit.raan = x[0][0]  
    orbit.aop  = x[1][0]  
    dV = x[2][0]

    # costs 1 extra run, but reruns using current X value.
    f_x = targetting_function(x, function_args)
    y_d_current = getattr(targetting_function, '_last_target', y_d)
    error = (f_x-y_d_current)
    return x,f_x,error

x ,f_x, error = differential_correction(
    x0 = np.array([(earth_parking.raan.value), (earth_parking.aop.value), 3.55]).reshape(3, 1),
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
print(f"\nParking Orbit RAAN = {np.rad2deg(x[0][0])} deg  | Parking Orbit AOP = {np.rad2deg(x[1][0])} deg | dV = {x[2][0]} km/s\n")
print(f'Heliocentric Position of Spacecraft at Mars arrival TOF: {f_x.flatten()} km with error of {error.flatten()} km compared to the target position {r2_mars.flatten()} km \n')
dV = x[2][0]

# '''
# USE THESE FOR above funciton call to fast track work
# iteration 6
# [CONVERGED] ERROR:[-7.94380903e-05 -7.21514225e-05 -3.13818455e-05]
# Parking Orbit RAAN = 91.12438014889646 deg  | Parking Orbit AOP = 265.06900325166424 deg | dV = 3.6365642436654273 km/s'''

# x , f_x, error = differential_correction(
#     x0 = np.array([np.deg2rad(91.12438014889646), np.deg2rad(265.06900325166424), 3.6365642436654273]).reshape(3, 1),
#     y_d = r2_mars.reshape(3,1),
#     targetting_function = mars_position_target_function,
#     function_args = (earth_parking, departure_date, arrival_date),
#     step_sizes = np.array([np.deg2rad(.01), np.deg2rad(.01), .01]).reshape(3, 1),
#     tol = np.array([10e-4, 10e-4, 10e-4]).reshape(3, 1),
#     max_i = 50,
#     orbit = earth_parking
# )
# print(f"\nParking Orbit RAAN = {np.rad2deg(x[0][0])} deg  | Parking Orbit AOP = {np.rad2deg(x[1][0])} deg | dV = {x[2][0]} km/s\n")
# print(f'Heliocentric Position of Spacecraft at Mars arrival TOF: {f_x.flatten()} km with error of {error.flatten()} km compared to the target position {r2_mars.flatten()} km \n')
# dV = x[2][0]
# --------------------------------------------------------------------------------------Step 5: Differential Correction: Mars B-Plane Targettings---------------------------------------------------------------------------------------
print("\n------------------------------------------------------------------------------------------------Phase 3: B-Plane Targetting ------------------------------------------------------------------------------------------------n")
# x = np.array([np.deg2rad(91.12437422103665), np.deg2rad(265.0703882273781), 3.6368109087080462]).reshape(3, 1)

def mars_pole_icrs(t):
    """
    Mars north pole unit vector in the ICRF/J2000 equatorial frame,
    from IAU WGCCRE rotational elements (Archinal et al., 2009/2015).
    """
    T = (t.tdb.jd - 2451545.0) / 36525.0  # Julian centuries from J2000 TDB
    alpha_0 = np.deg2rad(317.68143 - 0.1061 * T)   # pole RA
    delta_0 = np.deg2rad(52.88650 - 0.0609 * T)    # pole Dec
    N = np.array([ 
        np.cos(delta_0) * np.cos(alpha_0),
        np.cos(delta_0) * np.sin(alpha_0),
        np.sin(delta_0)
    ])
    #  This is exactly the same conversion you'd use to turn any RA/Dec sky position into an ICRF unit vector 
    return N / np.linalg.norm(N)

def mars_equatorial_frame(t):
    """R such that v_mars_eq = R @ v_icrf. Local Z = Mars pole, local X = ICRF∩Mars-equator node."""
    N_mars = mars_pole_icrs(t) # --> this is the Z axis of mars equatorial frame in icrf coords
    Z_icrf = np.array([0.0, 0.0, 1.0])
    # standards conventoin for any equatorial frame is that the x axs points along the intersection of the equatorial plane with the reference plane. here icrf equatorial 
    # intersection line 2 planes is cross product of the 2 normals. 
    x_axis = np.cross(Z_icrf, N_mars); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(N_mars, x_axis)
    # This is the standard direction-cosine-matrix fact: if the rows of R are the new frame's basis vectors written in the old frame's coordinates, then R*v projects V ontp each new basis vector in turn. 
    # i.e gives you v's components in the new frame. 
    return np.vstack([x_axis, y_axis, N_mars]) # Stacking the three new basis vectors as rows

def solve_achievable_plane(S_hat, inc_desired, t, branch=+1):
    """
    inc_desired is now interpreted relative to Mars' equator (its rotation pole),
    not ICRS Z. t is used to evaluate Mars' pole orientation (use t_soi -- Mars'
    pole barely precesses over a single mission, so exact epoch doesn't matter much).
    """
    R = mars_equatorial_frame(t)
    N_mars = R[2]

    cos_gamma = np.clip(np.dot(S_hat, N_mars), -1.0, 1.0)
    gamma = np.arccos(cos_gamma)
    if np.sin(gamma) < 1e-8:
        raise ValueError(f"Degenerate geometry: S_hat nearly parallel to Mars pole (gamma={np.degrees(gamma):.4f} deg)")
    inc_min = abs(np.pi/2 - gamma)
    inc_max = min(np.pi, np.pi/2 + gamma)
    inc_clamped = np.clip(inc_desired, inc_min, inc_max)

    # these 3 basis vectors are in plane perpendicualr to S_hat 
    # u1, u2, and phi — a geometric parameterization where inc_desired only picks which point on the achievable circle (phi) you land on
    u1 = N_mars - np.dot(N_mars, S_hat) * S_hat
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(S_hat, u1)

    cos_phi = np.clip(np.cos(inc_clamped) / np.sin(gamma), -1.0, 1.0)
    phi = branch * np.arccos(cos_phi)
    h_hat = np.cos(phi) * u1 + np.sin(phi) * u2

    h_eq = R @ h_hat  # rotated into Mars-equatorial coords
    inc_actual = np.arccos(np.clip(h_eq[2], -1.0, 1.0))
    raan = np.arctan2(h_eq[0], -h_eq[1]) % (2*np.pi)

    # these raan and inc are wrt to the icrs frame, but now in the mars equatoroal frame.
        # --> are meaningful now for mars parking orbit since now wrt to mars equator and poles
    return inc_actual, raan, h_hat, gamma, (inc_min, inc_max)

def bplane_target_function(x, fun_args):
    orbit, earth_soi, mars_soi, departure_date, arrival_date, mars_parking, inc_desired, branch = fun_args
    orbit.raan = x[0][0]
    orbit.aop  = x[1][0]
    dV = x[2][0]

    r_eci, v_eci = orbit_to_inertial_state(orbit)
    v_postburn_eci = v_eci + (dV * (v_eci/np.linalg.norm(v_eci)))
    Nbody_prop = variable_nbody_propagtion(r_eci, v_postburn_eci, earth_soi, mars_soi, departure_date, arrival_date)
    bplane_target_function._last_full_trajectory = Nbody_prop  # added for plotting
    periapsis_state = Nbody_prop['Leg 3 Mars Central']['periapsis']
    if periapsis_state is None:
        print("WARNING: periapsis not detected")
        bplane_target_function._last_TOF = None
        bplane_target_function._last_periapsis = None
        bplane_target_function._last_target = np.array([1e9, 1e9]).reshape(2, 1)
        return np.array([1e9, 1e9]).reshape(2, 1)
    
    r_periapsis, v_periapsis, t_periapsis = periapsis_state

    try: 
        # changed code here to now get the actual S_hat from the leg 2 SOI crossing state instead of using the lambert solution S_hat from vinf
        leg2 = Nbody_prop['Leg 2 Heliocentric']
        r_helio_soi = leg2['r'][-1]
        v_helio_soi = leg2['v'][-1]
        t_soi = leg2['t'][-1]
        r_mars_soi, v_mars_soi = get_body_barycentric_posvel('mars', t_soi)
        r_sun_soi, v_sun_soi = get_body_barycentric_posvel('sun', t_soi)
        r_mars_helio = (r_mars_soi.xyz - r_sun_soi.xyz).to(u.km).value
        v_mars_helio = (v_mars_soi.xyz - v_sun_soi.xyz).to(u.km/u.s).value
        vinf_actual_vec = v_helio_soi - v_mars_helio
        vinf_actual_mag = np.linalg.norm(vinf_actual_vec)
        S_hat = vinf_actual_vec / vinf_actual_mag

        inc_actual, raan_actual, h_hat, gamma, (inc_min, inc_max) = solve_achievable_plane(
        S_hat, inc_desired, t_soi, branch=branch    
        )
        # keep mars_parking's plane in sync with what's actually achievable this iteration
        mars_parking.inc = inc_actual
        mars_parking.raan = raan_actual

        B_hat = np.cross(S_hat, h_hat)
        B_hat /= np.linalg.norm(B_hat)

        rp_target = mars_parking.a.value * (1 - mars_parking.e.value)
        B_mag = rp_target * np.sqrt(1 + (2*MARS_MU.value)/(rp_target * vinf_actual_mag**2))
        N = np.array([0.0, 0.0, 1.0]) # B-plane T/R still ICRF Z -- unrelated to Mars pole, unchanged
        T_raw = np.cross(S_hat, N)
        T_norm = np.linalg.norm(T_raw)
        if T_norm < 1e-8:
            raise ValueError(f"Degenerate geometry: S_hat nearly parallel to ICRF Z (T-axis undefined)")
        T_hat = T_raw / T_norm
        R_hat = np.cross(S_hat, T_hat)
        R_hat = np.cross(S_hat, T_hat)
        BR_target = B_mag * np.dot(B_hat, R_hat)
        BT_target = B_mag * np.dot(B_hat, T_hat)

        # actual B-plane result of this trial trajectory, same frame
        rp, B_theta, BR, BT = Bplane2(r_periapsis, v_periapsis, MARS_MU.value)

    except ValueError as e:
        print(f"WARNING: invalid B-plane state ({e})")
        bplane_target_function._last_TOF = None
        bplane_target_function._last_periapsis = None
        bplane_target_function._last_target = np.array([1e9, 1e9]).reshape(2, 1)
        return np.array([1e9, 1e9]).reshape(2, 1)
    
    TOF_actual = (t_periapsis - departure_date).to_value('jd')

    # TOF is a diagnostic now, not a residual target
    bplane_target_function._last_TOF = TOF_actual
    bplane_target_function._last_periapsis = (r_periapsis, v_periapsis, t_periapsis)
    bplane_target_function._last_target = np.array([BR_target, BT_target]).reshape(2, 1)

    return np.array([BR, BT]).reshape(2, 1)


inc_desired = np.deg2rad(93.0)   # whatever you actually want, checked against the achievable band
branch = +1

# actual code to run - 08/13/2026
x, f_x, error = differential_correction(
    x0 = np.array([x[0][0], x[1][0], x[2][0]]).reshape(3, 1),
    y_d = None,  # see note below — target now floats each call
    targetting_function = bplane_target_function,
    function_args = (earth_parking, earth_soi, mars_soi, departure_date, arrival_date,
                      mars_parking, inc_desired, branch),
    step_sizes = np.array([np.deg2rad(.01), np.deg2rad(.01), .001]).reshape(3, 1),
    tol = np.array([.001, .001]).reshape(2, 1),
    max_i = 50,
    orbit = earth_parking)
dV = x[2][0]

# Shortcut to already converged
# x, f_x, error = differential_correction(
#     x0 = np.array([np.deg2rad(91.68002603420462), np.deg2rad(265.3348837832464), 3.64154769]).reshape(3, 1),
#     y_d = None,  # see note below — target now floats each call
#     targetting_function = bplane_target_function,
#     function_args = (earth_parking, earth_soi, mars_soi, departure_date, arrival_date,
#                       mars_parking, inc_desired, branch),
#     step_sizes = np.array([np.deg2rad(.01), np.deg2rad(.01), .001]).reshape(3, 1),
#     tol = np.array([.001, .001]).reshape(2, 1),
#     max_i = 50,
#     orbit = earth_parking)
# dV = x[2][0]

# results from nonshortcut code b4 fixing diff eq 08/17
'''[1] BR error: 4181.756605 km | BT error: -29797.896364 km | TOF (floating): 321.143 days
    Computed: BR=11849.273236 km, BT=-28314.998451 km
    Target:   BR=7667.516631 km, BT=1482.897913 km
    RAAN=91.775541 deg | AOP=265.375756 deg | dV=3.642255 km/s
    Mars Parking Orbit --> inc=93.000000 deg | raan=112.081161 deg
    
    [11] BR error: 0.000264 km | BT error: -0.000761 km | TOF (floating): 320.700 days
    Computed: BR=7670.241817 km, BT=1499.531614 km
    Target:   BR=7670.241553 km, BT=1499.532375 km
    RAAN=91.680027 deg | AOP=265.334882 deg | dV=3.641548 km/s
    Mars Parking Orbit --> inc=93.000000 deg | raan=112.274300 deg

    Mars miss distance at periapsis: 3748.027 km | Velocity at periapsis: 5.435809 km/s

[CONVERGED] ERROR:[ 0.0002643  -0.00076085]'''


# result from shortcut code above b4 fixing diff eq 08/17
'''[3] BR error: -0.000288 km | BT error: 0.000815 km | TOF (floating): 320.700 days
    Computed: BR=7670.241394 km, BT=1499.533098 km
    Target:   BR=7670.241682 km, BT=1499.532283 km
    RAAN=91.6800260216 deg | AOP=265.3348838024 deg | dV=3.6415476865 km/s
    Mars Parking Orbit --> inc=93.0000000000 deg | raan=112.2742990547 deg

    Mars miss distance at periapsis: 3747.986 km | Velocity at periapsis: 5.435832 km/s

[CONVERGED] ERROR:[-0.00028783  0.00081502]'''



"""  S_hat is fixed by the interplanetary trajectory - the simple lambert soln and departure state already determine is b4 touching bplane. 
    it lies in the hyperbolic orbital plane - h_hat dot S_hat = 0 

" You don't get to pick an arbitrary (inc, raan) for mars_parking and hit it with a single periapsis burn 
    --> you only get to pick a plane out of the one-parameter family of planes that contain S_hat. 
            That family is parameterized exactly by your B-plane angle, B_theta (rotating the B-vector around S_hat).

            So there are really two valid ways to close this loop:

1. Design mars_parking to be compatible — pick its (inc, raan) from the family of planes containing S_hat, rather than a priori. 
This is what real missions do: the arrival asymptote mostly dictates the achievable orbit plane unless you're willing to pay for a plane-change ΔV or bend the trajectory with a TCM upstream.
    --> S_hat is fixed on the arrival asymptote geometry. Any achieveable orbit plane nomrmal must be orthogonal to S_hat. i.e dot(h,S_hat) = 0. There exists several planes that allow for this. 
        --> N = Mars north poles and Gamma Y = angle between S_hat and N. the achieveable inclination sweeps as [ 90 - Gamma Y , 90 + gamma Y] 
                --> if S_hat is near the north pole, inclination is near polar. 
                    if S_hat is near equatorial palne, can hit almost any inclination ]
    
                    Pick an achieveable inclnation inside that band --> solve for matching inc/raan pair, set mars_parking to that pair, then target those B plane parameters. 

2. Solve for B_theta given a desired plane, and accept that if your chosen (inc, raan) doesn't satisfy ĥ·Ŝ=0, you'll get a best-fit (minimum-plane-change) solution — not an exact match. 
orth explicitly computing the achievable-vs-desired misalignment so you know how much of a plane-change burn you're implicitly assuming away.
    --> this one is kinda BS bc ur not really solving for an optimal parking orbit for ur flight path. ur just picking a random orbit and then trying to target it with a single periapsis burn.
        so you'd break apart ur angular momentum vector h_hat into component along s_hat and orthogonal to s_hat. 
            --> the component along s_hat is the part that you can't change with a single periapsis burn. the component orthogonal to s_hat is the part that you can change with a single periapsis burn.
                So the solution to this? You just trim off the part thats along s_hat. h = dot(h,S_hat)*S_hat + (h-(dot(h,S_hat)*S_hat))
                    --> so your angular momentum now is h = h - dot(h,S_hat)*S_hat. Using this, compute b plane parameters. 
                The mismatch between the actual angular momentum and the trimmed version is the angle between desired plane vs plane we're actually on.
                this angle i the miniimum plane change you'd have to do to get into desired orbit. the angle is the min plane change your assuming by targgeting the br and bt from the trimmed h. wuithout doing a plane change burn. 
                --> its "absorbing" a manuever cost that wasn't accounted for. """ 


""" side notes 08/06/2026 From NASA guide to interplanetary trasnfers

For flybys: 
DAP  - planetary equatorial declination of incoming asymptote, of Vinf, gives measure of minimum possible inclination of flyby. 
    --> negative of so is the latitude of vertical impact. 

mag fof vinf enables one to control flyby turn angle -Delta_psi between incoming and outgoing vinf vectors by choice of cloest approach
    --> delta_psi = 180-2*rho : comes from cos(rho) = 1/e = 1/ (1 + (vinf^2 * rp)/mu) 
    --> using vis visa, planetocntric vel at any time : V = sqrt((2*mu/r) + vinf^2)

ZAPS( zero angle periapsis sun) & ZAPE (zero angle periapsis earth) are angles between vinf and planet to sun & planet to earth respectfully
    --> map out the cone angle of the plaent furing the far encoutner phase for a sun or an earth oriented space crafpt. 
        --> ZAPS determines the phase angle of the planets solar illumination on the far encounter leg: phi_s = 180-ZAPS
    
The flyby is aimed at a point on the arrival target plane (B plane) --> plane passing throuhgh the planets centerand normal to Vinf 
    --> the asypotte goes through a point on B plane defined by BR and BT, where T and R axis are both orthogonal to Vinf. 
    Direction angle (B_theta) of B vectors measured CCW from T-axos to the Bvectors--> can be measured from incliantion of flyby orbit given from planet equatorial declination. 
        cos(inc) = cos(B_theta) * cos(DAP) -- > measured in the planet equatorial frame.

IN PLANET EQUATORIAL FRAME cords: 
right ascention of Mars ecliptic pole = 267.6227 deg
delcination of ecliptic pole = 63.2838 deg """


''' - some notes for myself: the B plane is a plane orthogonal/normal to the hyperbolic trajectory plane ( the incoming asymptote) and the initial hyperbolic excess velocity vector. 
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

    For both, need to extract the point wherein the n-body propagtor outputs the position & velocity of the s/c when entering MARS SOi'''

def capture_burn(r_periapsis, v_periapsis, mars_parking, mars_mu):
    """
    Single-impulse periapsis MOI burn. Assumes the arrival hyperbola and target
    parking orbit share periapsis radius, inclination, and RAAN (true if B-plane
    targeting converged with the fixed inc/raan pipeline) -- so the burn is purely
    tangential, aligned with -v_hat.
    """
    rp = np.linalg.norm(r_periapsis)
    v_hyp = np.linalg.norm(v_periapsis)
    v_hat = v_periapsis / v_hyp

    a_target = mars_parking.a.value
    v_target = np.sqrt(mars_mu * (2/rp - 1/a_target))   # vis-viva at same rp

    dV_mag = v_hyp - v_target          # retrograde burn magnitude
    dV_vec = -dV_mag * v_hat

    print(f"Periapsis radius:                    {rp:.3f} km")
    print(f"Hyperbolic arrival speed at rp:       {v_hyp:.6f} km/s")
    print(f"Target parking-orbit speed at rp:     {v_target:.6f} km/s")
    print(f"MOI capture burn magnitude:           {dV_mag:.6f} km/s")
    print(f"MOI capture burn vector (MCI):        {dV_vec} km/s")
    return dV_mag, dV_vec

r_peri, v_peri, t_peri = bplane_target_function._last_periapsis
dV_MOI, dV_MOI_vec = capture_burn(r_peri, v_peri, mars_parking, MARS_MU.value)

def propagate_two_body(r0,v0,mu,t0,dt = 60, parking_orbit = 'Earth'):
    r_mag = np.linalg.norm(r0)
    v_mag = np.linalg.norm(v0)
    a = -mu/(2*(.5*v_mag**2 - (mu/r_mag)))
    T = 2*np.pi * np.sqrt(a**3/mu)
    n_steps = int(T / dt)
    dt = TimeDelta(dt,format='sec')


    if parking_orbit == 'Mars':
        t0 = t0
    elif parking_orbit == 'Earth':
        t0 = t0 - TimeDelta(T, format='sec')
    else:
        raise ValueError("parking_orbit must be 'Earth' or 'Mars'")
    
    y = np.concatenate((r0, v0))
    t = t0
    r_list, v_list, t_list = [r0.copy()], [v0.copy()], [t0]
    for i in range(n_steps):
        y = RK4_single_step(y_dot_2body_earth, dt, t, y, mu)
        t = t + dt
        r_list.append(y[:3].copy()) # r = y[:3]
        v_list.append(y[3:].copy()) # v = y[3:6]
        t_list.append(t)
    return np.array(r_list), np.array(v_list), t_list , a ,T

def subsample_indices(n, target=300):
    """Keeps animation frame counts and pickle size manageable."""
    if n <= target:
        return np.arange(n)
    stride = max(1, n // target)
    idx = np.arange(0, n, stride)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx

def times_to_jd(t_list):
    return np.array([t.jd for t in t_list])

# --- 1. Earth parking orbit: construct 1 period before departure, propagate forward TO departure ---
r_eci_preburn, v_eci_preburn = orbit_to_inertial_state(earth_parking)  # state AT departure (burn point)
# propagating forward one full period returns to this exact state at departure_date
r_earthpark_full, v_earthpark_full, t_earthpark_full, a_earthpark, T_earthpark = propagate_two_body(
    r_eci_preburn, v_eci_preburn, EARTH_MU.value, departure_date, dt=30, parking_orbit ='Earth')

idx_ep = subsample_indices(len(r_earthpark_full))   # e.g. array([0, 6, 12, ..., 179])
r_earthpark = r_earthpark_full[idx_ep]              # pick those rows out of the position array
t_earthpark = [t_earthpark_full[i] for i in idx_ep] # pick the matching timestamps (list, not array, since these are astropy Time objects)

# --- 2. Leg 1: Earth-centered departure (post-burn)
traj = bplane_target_function._last_full_trajectory
leg1_r_full = np.array(traj['Leg 1 Earth Central']['r'])
leg1_t_full = traj['Leg 1 Earth Central']['t']
idx1 = subsample_indices(len(leg1_r_full))
leg1_r = leg1_r_full[idx1]
leg1_t = [leg1_t_full[i] for i in idx1]

# --- 3. Leg 2: Heliocentric Earth->Mars
leg2_r_full = np.array(traj['Leg 2 Heliocentric']['r'])
leg2_t_full = traj['Leg 2 Heliocentric']['t']
idx2 = subsample_indices(len(leg2_r_full))
leg2_r = leg2_r_full[idx2]
leg2_t = [leg2_t_full[i] for i in idx2]

# --- 4. Leg 3: Mars-centered incoming
leg3_r_full = np.array(traj['Leg 3 Mars Central']['r'])
leg3_t_full = traj['Leg 3 Mars Central']['t']
idx3 = subsample_indices(len(leg3_r_full))
leg3_r = leg3_r_full[idx3]
leg3_t = [leg3_t_full[i] for i in idx3]

# --- Earth & Mars heliocentric position tracks, sampled at Leg 2's times ---
# (needed so the heliocentric plot can show Earth/Mars moving during transfer)
earth_track_r = []
mars_track_r = []
for t_local in leg2_t:
    r_earth_b, _ = get_body_barycentric_posvel('earth', t_local)
    r_mars_b, _  = get_body_barycentric_posvel('mars', t_local)
    r_sun_b, _   = get_body_barycentric_posvel('sun', t_local)
    earth_track_r.append((r_earth_b.xyz - r_sun_b.xyz).to(u.km).value)
    mars_track_r.append((r_mars_b.xyz - r_sun_b.xyz).to(u.km).value)
earth_track_r = np.array(earth_track_r)
mars_track_r = np.array(mars_track_r)

# --- 5. Post-MOI capture orbit, 1 period after burn
v_post_burn = v_peri + dV_MOI_vec
r_park_full, v_park_full, t_park_full, a_park, T_park = propagate_two_body(
    r_peri, v_post_burn, MARS_MU.value, t_peri, dt=30, parking_orbit ='Mars')

idx_cap = subsample_indices(len(r_park_full))
r_park = r_park_full[idx_cap]
t_park = [t_park_full[i] for i in idx_cap]

# --- Save everything to disk ---
import pickle

mission_data = {
    'earth_parking': {'r': r_earthpark, 't_jd': times_to_jd(t_earthpark),
                       'a': a_earthpark, 'T': T_earthpark},
    'leg1_earth_centered': {'r': leg1_r, 't_jd': times_to_jd(leg1_t)},
    'leg2_heliocentric':   {'r': leg2_r, 't_jd': times_to_jd(leg2_t)},
    'earth_track': {'r': earth_track_r, 't_jd': times_to_jd(leg2_t)},
    'mars_track':  {'r': mars_track_r, 't_jd': times_to_jd(leg2_t)},
    'leg3_mars_centered':  {'r': leg3_r, 't_jd': times_to_jd(leg3_t)},
    'capture_orbit': {'r': r_park, 't_jd': times_to_jd(t_park),
                       'a': a_park, 'T': T_park},
    'key_points': {
        'r_peri': r_peri, 'v_peri': v_peri, 't_peri_jd': t_peri.jd,
        'dV_MOI_vec': dV_MOI_vec, 'dV_MOI_mag': dV_MOI,
        'earth_soi': earth_soi, 'mars_soi': mars_soi,
        'departure_date_jd': departure_date.jd,
    }
}

with open('mission_data.pkl', 'wb') as f:
    pickle.dump(mission_data, f)

print("Saved mission_data.pkl")

code = 1