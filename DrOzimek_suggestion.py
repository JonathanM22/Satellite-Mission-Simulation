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

# Whole damn function written out by VSCODE

# def newton_raphson_ephem(mu, r0, v0, dt, t0, tol=1e-8, max_iter=1000, fun_arg: list=[]):
#     """
#     Universal Variable Formulation with Ephemeris Data
#     """

#     # Initial guess for chi
#     r0_mag = np.linalg.norm(r0)
#     v0_mag = np.linalg.norm(v0)
#     alpha = 2/r0_mag - (v0_mag**2)/mu
#     chi = np.sqrt(mu) * abs(alpha) * dt

#     # Newton-Raphson Iteration
#     for i in range(max_iter):
#         z = alpha * chi**2
#         S, C = stumpff_functions(z)

#         r = (chi**2 * C) + (np.dot(r0, v0)/np.sqrt(mu)) * chi * (1 - z * S) + r0_mag * (1 - z * C)
#         f = (r - np.sqrt(mu) * dt)

#         if abs(f) < tol:
#             break

#         df_dchi = (chi * (1 - z * S)) + (np.dot(r0, v0)/np.sqrt(mu)) * (1 - z * C)

#         chi = chi - f / df_dchi

#     # Compute final position and velocity vectors
#     f = 1 - (chi**2 / r0_mag) * C
#     g = dt - (1/np.sqrt(mu)) * chi**3 * S

#     r_vec = f * r0 + g * v0

#     fdot = (np.sqrt(mu) / (r0_mag * np.linalg.norm(r_vec))) * chi * (z * S - 1)
#     gdot = 1 - (chi**2 / np.linalg.norm(r_vec)) * C

#     v_vec = fdot * r0 + gdot * v0

#     return r_vec, v_vec

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
    Vinf_values = np.array([r['Vinf_arrival'] for r in results])
    Vinf_mag = np.linalg.norm(Vinf_values, axis=1)

    # tof_values = np.array([r['tof_days'] for r in results])

    # normalzing them to be between 0 and 1 
    C3_norm = (C3_values - np.min(C3_values)) / (np.max(C3_values) - np.min(C3_values))
    Vinf_norm = (Vinf_mag - np.min(Vinf_mag)) / (np.max(Vinf_mag) - np.min(Vinf_mag))

    # give weighted scoe
    score = (weight_C3 * C3_norm) + (weight_Vinf * Vinf_norm)
    # minimize
    optimal_idx = np.argmin(score)
    optimal_C3 = C3_values[optimal_idx]
    optimal_Vinf = Vinf_values[optimal_idx]
    optimal_solution = [optimal_C3, optimal_Vinf]

    print(f"\nOptimal Mission Duration: {results[optimal_idx]['tof_days']} Days. Arrival Date = {departure_date+results[optimal_idx]['tof_days']} with (C3: {optimal_solution[0]:.3f} km²/s², Vinf Arrival: {Vinf_mag[optimal_idx]:.3f}) km/s\n")
    return optimal_solution

# outputs array of optimal C3 & Vinf arrival based on assigned weights ( user defined )
optimal_solution = find_optimal_solution(results, weight_C3=0.75, weight_Vinf=0.25)
C3 = optimal_solution[0]
Vinf_arrival = optimal_solution[1]*(u.km/u.s)
'''
at this point I can differ my approach. 
1. min good balance between c3 and vinf arrival --> convert to raan/dec/inclination for launch window --> continue with what Dr Ozimek suggested
2. Select best C3 and vinf arrival --> full n-body simulaiton from Earth and Mars using rk4 & ydot_n_ephemeris --> 
    diff between r2_mars & final pos form n-body sim to see how close we are to mars --> 
        adjust initial vel accordingly and re run n-body sim until close enough to mars (for some cap) --> 
           B-plane targetting
'''

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

def vinf_to_raan_dec(Vinf): 

    Dec = np.arcsin(Vinf[2]/np.linalg.norm(Vinf))
    RAAN = np.arctan2(Vinf[1],Vinf[0])
    # print(f'RAAN: {np.degrees(RAAN): .3f}° | Declination: {np.degrees(Dec): .3f}°')

    return RAAN, Dec

RAAN_dep, Dec_dep = vinf_to_raan_dec(optimal_solution[1]) 
print(f'Outbound RAAN = {np.degrees(RAAN_dep): .3f}° | Outbound Declination = {np.degrees(Dec_dep): .3f}°\n')

Earth_rad = 6378 #km


# Define Earth Parking Orbit

earth_parking = Orbit(mu=EARTH_MU,
                      a=32000*u.km,
                      e=0.80*u.km/u.km,  # unitless
                      f0=(0*u.deg).to(u.rad),
                      inc=(28.5*u.deg).to(u.rad),
                      raan=(175*u.deg).to(u.rad),
                      aop=(240*u.deg).to(u.rad)
                      )

# this function creates the X vector of Parking Orbit RAAN and AOP: the independent variables, that we will adjust to minimize the error vector
# --> want to adjust those 2 until our outbound asymptote matches the lamert soln vinf raan and dec

def sat_orbit_targeting(orbit, v_inf_mag, x):

    # orbit: Orbit object
    # defined this say so when using newtons methods, we can just pass in the orbit object and modify the raan and aop values directly
    orbit.raan = (x[0][0]*u.deg).to(u.rad)
    orbit.aop = (x[1][0]*u.deg).to(u.rad)

    # Getting intial position & vel. This r is wrt to the central body: earth
    r = orbit.r_at_true_anomaly(orbit.f0)
    v = np.sqrt(2*(orbit.energy + (orbit.mu/r)))

    # hyperbolic velocity at perigee
    v_hyp = np.sqrt(2*(((v_inf_mag**2)/2) + (orbit.mu/r)))

    # Converts the orbital elements from the orbital frame to the perifocal frame
    r_pqw, v_pqw = orb_2_pqw(r.value,
                             orbit.f0.value, orbit.e.value,
                             orbit.p.value, orbit.mu.value)
    
    # converts perifocal frame to eci frame
    r_eci, v_eci = perif_2_eci(r_pqw, v_pqw, orbit.inc.value,
                               orbit.raan.value,
                               orbit.aop.value)

    # direction of delta V --> tangential to orbit --> same direction as v eci
    sat_v_dir = v_eci/np.linalg.norm(v_eci)

    # delta V that has to get applied to the satellite: Dv = V,p_hyp - V (in LEO using vis-viva)
    dv = v_hyp - v

    # New velocity after applying delta V. same as the hyperbolic excess velocity
    v_eci = (v_eci + dv.value*sat_v_dir)*(u.km/u.s) 

    raan, dec = vinf_to_raan_dec(v_eci)

    return np.array([raan.value, dec.value]).reshape(2, 1)

# Finite difference sensitivity matrix for newtons method. 
# --> outputs the (2) columns of the jacobian matrix of the partial derivatives: d(raan,dec)/d(raan,aop) where RAAN and AOP are parking orbit independent variables and raan and dec are the outbound asymptote dependent variables
def sensitivity_matrix(orbit, v_inf_mag, x, dt_raan, dt_aop):

    # Reshape dt_input args into dt vectors
    dt_rann_ar = np.array([dt_raan, 0]).reshape(2, 1)
    dt_aop_ar = np.array([0, dt_aop]).reshape(2, 1)

    # equations from AGI newtons method paper 
    dt_raan_col = (1/dt_raan)*(sat_orbit_targeting(
        orbit, v_inf_mag, x + dt_rann_ar) - sat_orbit_targeting(orbit, v_inf_mag, x))

    dt_aop_col = (1/dt_aop)*(sat_orbit_targeting(
        orbit, v_inf_mag, x + dt_aop_ar) - sat_orbit_targeting(orbit, v_inf_mag, x))

    return np.block([dt_raan_col, dt_aop_col])

Vinf_mag = np.linalg.norm(Vinf_arrival).to(u.km/u.s)
vinf_raan, vinf_dec = vinf_to_raan_dec(Vinf_arrival)

raan0 = 175
aop0 = 240
x0 = np.array([raan0, aop0]).reshape(2, 1)
y0 = sat_orbit_targeting(earth_parking, Vinf_mag, x0)
dt_raan = np.deg2rad(45)
dt_aop = np.deg2rad(180)

i = 0
max_i = 20000
y_d = np.array([vinf_raan.value, vinf_dec.value]).reshape(2, 1)
x = x0
tol = np.array([10e-6, 10e-6]).reshape(2, 1)
error = y0 - y_d

while np.any(np.abs(error) > tol):
    f_x = sat_orbit_targeting(earth_parking, Vinf_mag, x)
    J = sensitivity_matrix(earth_parking, Vinf_mag, x, dt_raan, dt_aop)

    x_k = x - np.linalg.inv(J)@(f_x-y_d)
    f_xk = sat_orbit_targeting(earth_parking, Vinf_mag, x_k)
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

f_x = sat_orbit_targeting(earth_parking, Vinf_mag, x)
error = (f_x-y_d)
print(f"x | earth.raan = {x[0][0]}, earth.aop = {x[1][0]}")
print(f"SatVel@f | raan: {f_x[0][0]} rad | dec: {f_x[1][0]} rad")
print(f"V_inf | raan: {y_d[0][0]} rad | dec: {y_d[1][0]} rad")


# ------------------------------------------------------------------------------------------Gonna pause here and try to do n-body correction---------------------------------------------------------------------------------------------

# central_body = sun
# bodies = [mercury,venus,jupiter,saturn,uranus,neptune]
# fun_arg = [central_body,bodies]

# # _, _, ys = propagate_rk4(sat.r0.value, sat.v0.value, t0, tf, dt, fun_arg)
# dt = TimeDelta(3600, format='sec')
# r_sats, _, _ = propagate_rk4(r1_earth, transfer_v1, departure_date, arrival_date, dt, fun_arg=fun_arg)

# r_mars_miss = r_sats[-1] - r2_mars
# print(f'Satellite Missed Mars Target by {np.linalg.norm(r_mars_miss):.5f} km')

# # # --> work on diff eq corrector

# # # thinking of doing newton raphson with f and g functions --> iterate v(_,_,_) = 0 --> proceed normally

'''
# TLDR --> Satellite misses mars by 131768.52445 km
# propagate the new satellite initial conditions after adjusting RAAN and AOP
'''
# this pos vector is wrt earth center
r1_idealized = earth_parking.r_at_true_anomaly(earth_parking.f0).value
r_pqw, _ = orb_2_pqw(r1_idealized,
                            earth_parking.f0.value, earth_parking.e.value,
                            earth_parking.p.value, earth_parking.mu.value)

# converts perifocal frame to eci frame
r_eci, _ = perif_2_eci(r_pqw, _, earth_parking.inc.value,
                            earth_parking.raan.value,
                            earth_parking.aop.value)
                            

# Your satellite's heliocentric position
r1_sat_helio = r1_earth + r_eci  # Add Earth's position to your ECI position

# for redundancy since we can reuse our vinf dept vector, we can just recalc the departure velocity after adjusting RAAN and AOP 

def raan_dec_to_vinf(Vinf_mag, raan, dec):
    
    x = Vinf_mag*np.sin(np.pi/2-dec)*np.cos(raan)
    y = Vinf_mag*np.sin(np.pi/2-dec)*np.sin(raan)
    z = Vinf_mag*np.cos(np.pi/2-dec)
    return np.array([x,y,z])

Vinf_dept_idealized = raan_dec_to_vinf(Vinf_mag.value, x[0][0], x[1][0])
Vinf_ECI = Vinf_dept_idealized   # departure hyperbolic excess velcity in the ECI frame

central_body = sun
bodies = [mercury,venus,jupiter,saturn,uranus,neptune]
fun_arg = [central_body,bodies]

# propagate with new initial conditions from parking orbit targeting

# _, _, ys = propagate_rk4(sat.r0.value, sat.v0.value, t0, tf, dt, fun_arg)
dt = TimeDelta(3600, format='sec')
r_sats, _, _ = propagate_rk4(r1_sat_helio, Vinf_ECI, departure_date, arrival_date, dt, fun_arg=fun_arg)

r_mars_miss = r_sats[-1] - r2_mars
print(f'Satellite Missed Mars Target by {np.linalg.norm(r_mars_miss):.5f} km')

