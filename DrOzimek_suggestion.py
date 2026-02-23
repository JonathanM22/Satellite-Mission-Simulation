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
    y0 = np.concatenate((r0, v0))
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
    transfer_angle = (np.acos(np.dot(r1_earth, r2_mars) / (np.linalg.norm(r1_earth) * np.linalg.norm(r2_mars))))
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
    optimal_arrival_earth_r2 = r2_mars_vectors[optimal_idx]
    optimal_arrival_mars_v2 = v2_mars_vectors[optimal_idx]
    arrival_date = arrival_dates[optimal_idx]
    print(f"\nOptimal Mission Duration: {results[optimal_idx]['tof_days']} Days. Arrival Date = {[arrival_date]} with (C3: {optimal_C3:.3f} km²/s², Vinf Arrival: {np.linalg.norm(optimal_Vinf_arrival):.3f} km/s, Vinf Departure: {np.linalg.norm(optimal_Vinf_departure):.3f} km/s)\n")
    return optimal_C3, optimal_Vinf_departure, optimal_Vinf_arrival, optimal_transfer_v1, optimal_arrival_earth_r2,optimal_arrival_mars_v2, arrival_date

# outputs array of optimal C3 & Vinf arrival based on assigned weights ( user defined )
optimal_C3, optimal_Vinf_departure, optimal_Vinf_arrival, optimal_transfer_v1,r2_mars,v2_mars, arrival_date = find_optimal_solution(results, weight_C3=0.75, weight_Vinf=0.25)
C3 = optimal_C3

# V infinty departure and arrival vectors derived from lamberts: remember these are in the heliocentric ecliptic frame
Vinf_departure = optimal_Vinf_departure #*(u.km/u.s)
Vinf_arrival = optimal_Vinf_arrival #*(u.km/u.s)
transfer_v1 = optimal_transfer_v1

# V infinty arival and departure magnitudes 
Vinf_arrival_mag = np.linalg.norm(optimal_Vinf_arrival) #*(u.km/u.s)
Vinf_departure_mag = np.linalg.norm(optimal_Vinf_departure) #*(u.km/u.s)

# ------------------------------------------------------------------------------------------Gonna pause here and try to do n-body correction---------------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------------Step 2: Converting Vinf to RAAN/DEC/Inclination-------------------------------------------------------------------------------------------

# to get declination and RAAN from vinf vector, we need the velocity to be in the ECI frame --> alr is 

'''
Calculate lambert soln to get estimated C3, vinf arrival, RAAN, & dec
    --> Define arbritrary earth parking orbit 
    --> Parking orbit --> ECI frame pos and vel
    --> apply delta v to find v inf vector dept (C3)
    --> calculate Raan and v3 from vinf vector 
    --> create error vector between lamberts and parking orbit
    --> finite difference jacobian to adjust parking orbit elements (RAAN & AOP) to minimize error vector
    --> repeat, then propogate the final rsat_0 and vsat_0 with the nbody propagator using the rk4 funciton 
    --> gonna miss mars, but will implement differential correction for b plane targetting later
'''

'''
# THe outbound launch asympote direction is defined by the right ascension and declination of the asymptote vector, which we can get from the Vinf departure vector from lamberts. 
# We can then target the parking orbit to match that RAAN and declination, which will ensure that our outbound asymptote matches the lamberts soln vinf raan and dec.
'''
def vinf_to_raan_dec(Vinf): 

    Dec = np.arcsin(Vinf[2]/np.linalg.norm(Vinf))
    RAAN = np.arctan2(Vinf[1],Vinf[0])
    # print(f'RAAN: {np.degrees(RAAN): .3f}° | Declination: {np.degrees(Dec): .3f}°')

    return RAAN, Dec

# All heliocentric position and velocity vectors are in the ecliptic frame, so we need to convert the vinf departure vector to the ECI frame to get the correct RAAN and declination for the parking orbit targeting

# Convert V_infinity from ecliptic to ECI frame
earth_tilt = np.deg2rad(23.43928)

# Rotation matrix: ecliptic → ECI
ecliptic_to_eci = np.array([
    [1, 0, 0],
    [0, np.cos(earth_tilt), np.sin(earth_tilt)],
    [0, -np.sin(earth_tilt), np.cos(earth_tilt)]
])

earth_tilt = np.deg2rad(23.43928)  # difference between equatorial plane and ecliptic plane

eci_to_ecliptic = np.array([
    [1, 0, 0],
    [0, np.cos(earth_tilt), -np.sin(earth_tilt)],
    [0, np.sin(earth_tilt), np.cos(earth_tilt)]
])

Vinf_departure_eci = ecliptic_to_eci @ Vinf_departure
RAAN_dep, Dec_dep = vinf_to_raan_dec(Vinf_departure_eci) 
print(f'Outbound RAAN = {np.degrees(RAAN_dep): .3f}° | Outbound Declination = {np.degrees(Dec_dep): .3f}°\n')

# Earth_rad = 6378 #km

# Define Earth Parking Orbit
earth_parking = Orbit(mu=EARTH_MU,
                      a=(400+6378)*u.km,
                      e=0.0*u.km/u.km,  # unitless
                      f0=(0*u.deg).to(u.rad),
                      inc=(28.5*u.deg).to(u.rad),
                      raan=(175*u.deg).to(u.rad),
                      aop=(240*u.deg).to(u.rad)
                      )

# this function creates the X vector of Parking Orbit RAAN and AOP: the independent variables, that we will adjust to minimize the error vector
# --> want to adjust those 2 until our outbound asymptote matches the lamert soln vinf raan and dec

'''
We want to target obtianing a parking orbit such tha when we apply the delta V to get onto the hyperbolic escape trajectory, the resulting Vinf vector produces the same launch asymptote with the  direction (same RAAN and declination) as the Vinf vector from lamberts.
Might have to tweak where exactly we apply the delta V, but for now just applying it at periapsis and seeing what happens.

Thinking of creating an error matrix between the Vinf from lamberts and some b plane targeting Vinf that we get from the parking orbit, then using newtons method to adjust the RAAN and AOP of the parking orbit until we minimize the error between the lamberts Vinf and the parking orbit Vinf.
'''
# -----------------------------------------------------------------------------Step 3:Differential Correction: Finding Correct outbound asymptote direction-----------------------------------------------------------------------------

def sat_orbit_targeting(orbit, v_inf_mag, x):


    # orbit: Orbit object
    # defined this say so when using newtons methods, we can just pass in the orbit object and modify the raan and aop values directly
    orbit.raan = (x[0][0])
    orbit.aop = (x[1][0])

    # Converts the orbital elements from the orbital frame to the perifocal frame
    r_pqw, v_pqw = orb_2_pqw(orbit.r_at_true_anomaly(orbit.f0).value,
                             orbit.f0.value, orbit.e.value,
                             orbit.p.value, orbit.mu.value)
    
    # converts perifocal frame to eci frame
    r_eci, v_eci = perif_2_eci(r_pqw, v_pqw, orbit.inc.value,
                               orbit.raan,
                               orbit.aop)
    
     # hyperbolic velocity at perigee
    v_hyp = np.sqrt(2*(((v_inf_mag**2)/2) + (orbit.mu.value/np.linalg.norm(r_eci))))

    # direction of delta V --> tangential to orbit --> same direction as v eci
    sat_v_dir = v_eci/np.linalg.norm(v_eci)

    # delta V that has to get applied to the satellite: Dv = V,p_hyp - V (in LEO using vis-viva)
    # the instantaneous Delta V is applied to the periapsis of the parking orbit 
    dv = v_hyp - np.linalg.norm(v_eci)

    # v_postburn_eci = (v_eci + dv*sat_v_dir) 
    # v_inf = v_postburn_eci - v1_earth

    e_hyp = 1 + (np.linalg.norm(r_eci)*v_inf_mag**2)/(orbit.mu.value) # eccentricity of hyperbolic escape trajectory
    print(f'Eccentricity of hyperbolic escape trajectory: {e_hyp:.3f}')
    # rp and e point in same direction
    a_hyp = -orbit.mu.value/v_inf_mag**2 # semi major axis of hyperbolic escape trajectory
    e_hat = r_eci/np.linalg.norm(r_eci) # unit vector in direction of eccentricity vector, which points towards periapsis 
    h = np.cross(r_eci, v_eci) # specific angular momentum vector
    h_hat = h/np.linalg.norm(h) # unit vector in direction of specific angular momentum vector, which is perpendicular to the orbital plane
    t_hat = np.cross(h_hat, e_hat) # unit vector in direction of tangential velocity, which is perpendicular to both the eccentricity vector and the specific angular momentum vector

    # this vector is still in the perifocal frame. Need to change to ECI 
    s_hat = -1/e_hyp * e_hat - np.sqrt((1-1/e_hyp**2))*t_hat 
    # s_hat  = -1/e_hyp * e_hat - np.sqrt(1 - (orbit.mu.value**2/v_inf_mag**4*a_hyp**2*e_hyp**2))*t_hat 
    s_hat = perif_2_eci_DCM(orbit.inc.value, orbit.raan, orbit.aop) @ s_hat
    print(f'\n{s_hat}\n')
    v_inf = v_inf_mag * s_hat
    '''
    # ADDED THESE TO SEE POSSIBLE COMBINATIONS AND TO SEE IF ANY ALIGN WITH THE V INF DIRECTION 
    
    v_postburn_eci = v_eci + dv*sat_v_dir
    v_postburn_ecliptic = eci_to_ecliptic @ v_postburn_eci
    transfer_v1_helio = v_postburn_ecliptic + v1_earth
    transfer_v1_helio_eci = ecliptic_to_eci @ transfer_v1_helio
    maybe_vinf = eci_to_ecliptic @ v_postburn_ecliptic - v1_earth
    
    print(f'\n{transfer_v1_helio_eci/np.linalg.norm(transfer_v1_helio_eci)}\n') 
    print(f'\n{v_postburn_eci/np.linalg.norm(v_postburn_eci)}\n') 
    print(f'\n{maybe_vinf/np.linalg.norm(maybe_vinf)}\n') 

    '''
  
    raan, dec = vinf_to_raan_dec(v_inf)


    return np.array([raan, dec]).reshape(2, 1)
'''
 # this is straight up wrong. This is the post burn velocity after the spacecraft. which, funny enough, is just the hyperbolic perigee velocity. 
    # this is not the same as the vinf vector at all --> needs to be changed
   
    # v_eci = (v_eci + dv*sat_v_dir) 

    # wrong --> need the VINF vector, not the post burn. 
    # the post burn velocity vector is the one that we should be transforming into the ecliptic frame and propagting towards mars. THis is analgous to transfer_v1 from lambers. 

    
    # raan, dec = vinf_to_raan_dec(v_eci)
    
    
    # replace the v_eci above and actually solve for vinf vector. 
'''
   

# Finite difference sensitivity matrix for newtons method. 
# --> outputs the (2) columns of the jacobian matrix of the partial derivatives: d(raan,dec)/d(raan,aop) where RAAN and AOP are parking orbit independent variables and raan and dec are the outbound asymptote dependent variables
def sensitivity_matrix(orbit, v_inf_mag, x, dt_raan, dt_aop):

    # Reshape dt_input args into dt vectors
    dt_rann_ar = np.array([dt_raan, 0]).reshape(2, 1)
    dt_aop_ar = np.array([0, dt_aop]).reshape(2, 1)

    # equations from AGI newtons method paper 
    dt_raan_col = (1/(dt_raan))*(sat_orbit_targeting(
        orbit, v_inf_mag, x + dt_rann_ar) - sat_orbit_targeting(orbit, v_inf_mag, x))

    dt_aop_col = (1/(dt_aop))*(sat_orbit_targeting(
        orbit, v_inf_mag, x + dt_aop_ar) - sat_orbit_targeting(orbit, v_inf_mag, x))

    return np.block([dt_raan_col, dt_aop_col])

raan0 = np.deg2rad(175)
aop0 = np.deg2rad(240)
x0 = np.array([raan0, aop0]).reshape(2, 1)
y0 = sat_orbit_targeting(earth_parking, Vinf_departure_mag, x0)
dt_raan = np.deg2rad(.01)
dt_aop = np.deg2rad(.01)

i = 0
max_i = 20000
y_d = np.array([RAAN_dep, Dec_dep]).reshape(2, 1)
x = x0
tol = np.array([10e-8, 10e-8]).reshape(2, 1)
error = y0 - y_d

while np.any(np.abs(error) > tol):
    f_x = sat_orbit_targeting(earth_parking, Vinf_departure_mag, x)
    J = sensitivity_matrix(earth_parking, Vinf_departure_mag, x, dt_raan, dt_aop)

    x_k = x - np.linalg.inv(J)@(f_x-y_d)
    f_xk = sat_orbit_targeting(earth_parking, Vinf_departure_mag, x_k)
    error = (f_xk-y_d)

    dt = (x_k-x)*np.linalg.norm(error)

    dt_raan = dt[0][0]
    dt_aop = dt[1][0]

    print(f"[{i}] ERROR:{error.flatten()}| DT: {dt.flatten()}")

    x = x_k
    i += 1
    if i > max_i:
        print(f"[MAX ITER] ERROR:{error.flatten()}")
        break

if np.linalg.norm(error) < 0.1:
    print(f"===========================================")
    print(f"[TOL SATISFIED] ERROR:{error.flatten()}")
else:
    print(f"===========================================")
    print(f"[TOL NOT SATISFIED] ERROR:{error.flatten()}")

f_x = sat_orbit_targeting(earth_parking, Vinf_departure_mag, x)
error = (f_x-y_d)
print(f"x | earth.raan = {np.rad2deg(x[0][0])} deg  | earth.aop = {np.rad2deg(x[1][0])} deg ")
print(f"SatVel@f | raan: {f_x[0][0]} rad | dec: {f_x[1][0]} rad")
print(f"V_inf | raan: {y_d[0][0]} rad | dec: {y_d[1][0]} rad")

# ------------------------------------------------------------------------------------------Propagating Parking Orbit and Post Delta V Trajectory---------------------------------------------------------------------------------------------

# Update parking orbit with converged values
earth_parking.raan = ((x[0][0]))
earth_parking.aop = ((x[1][0]))

# this pos vector is wrt earth center
r1_earth_parking = earth_parking.r_at_true_anomaly(earth_parking.f0).value

r_pqw, v_pqw = orb_2_pqw(r1_earth_parking,
                            earth_parking.f0.value, earth_parking.e.value,
                            earth_parking.p.value, earth_parking.mu.value)

# converts perifocal frame to eci frame
r_eci, v_eci = perif_2_eci(r_pqw, v_pqw, earth_parking.inc.value,
                            earth_parking.raan,
                            earth_parking.aop)
                            
v_hyp = np.sqrt(2*(((Vinf_departure_mag**2)/2) + (earth_parking.mu.value/np.linalg.norm(r_eci))))
sat_v_dir = v_eci/np.linalg.norm(v_eci)
delta_v = v_hyp - np.linalg.norm(v_eci)
v_postburn_eci = (v_eci + delta_v*sat_v_dir)

'''
# # Your satellite's heliocentric position
# r1_sat_helio = r1_earth + r_eci  # Add Earth's position to your ECI position

THIS IS WRONG SINCE r_eci is in the ECI frame, wrt to earths equatorial plane. We need to convert r_eci to heliocentric ecliptic frame first
This is because Earth's axis is tilted at 23.5 degrees, so the ECI frame is not aligned with the heliocentric ecliptic frame. We need to rotate the r_eci vector by 23.5 degrees to get the correct heliocentric position of the satellite.

ECI FRAME: The X axis points towards vernal equinox, Z points towards earths north pole and y completes the set:
    ECI frame is tilted wrt to ecliptic frame by 23.5 degrees. The eci frame ignores earths tilt and xy plane is the equator 

Ecliptic Frame: The X axis points towards vernal equinox, Z axis is perpendicular to the ecliptic plane (plane of earths orbit around sun), and y completes the set
    ecliptic plane is the plane of earths orbit around the sun


SAME FOR VELOCITY
'''

r_sat_ecliptic = eci_to_ecliptic @ r_eci  # km
r1_sat_helio = r1_earth + r_sat_ecliptic

#Lambert’s geometry and reconstructing a different orbit that merely has the same asymptote angles. WRONG 

v_postburn_ecliptic = eci_to_ecliptic @ v_postburn_eci
Transfer_V1_idealized = v_postburn_ecliptic + v1_earth

Vinf_non_lamberts = Transfer_V1_idealized - v1_earth
print(f'Non Lambert Vinf: {np.linalg.norm(Vinf_non_lamberts):.3f} km/s)\n')
print(f'Lambert Vinf: {np.linalg.norm(Vinf_departure):.3f} km/s)\n')

central_body = sun
bodies = [mercury,venus,jupiter,saturn,uranus,neptune]
fun_arg = [central_body,bodies]

# propagate with new initial conditions from parking orbit targeting

# _, _, ys = propagate_rk4(sat.r0.value, sat.v0.value, t0, tf, dt, fun_arg)
dt = TimeDelta(3600, format='sec')
r_sats, _, _ = propagate_rk4(r1_sat_helio, Transfer_V1_idealized, departure_date, arrival_date, dt, fun_arg=fun_arg)

r_mars_miss = r_sats[-1] - r2_mars
print(f'Satellite Missed Mars Target by {np.linalg.norm(r_mars_miss):.5f} km')

