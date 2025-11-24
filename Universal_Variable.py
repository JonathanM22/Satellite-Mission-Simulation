import numpy as np
import math
from scipy import optimize


# Algorithm 1: Computation of Stumpff Functions C2 and C3
# set C2(psi) = ( 1 - cos(sqrt(psi)) ) / psi)
# set C3(psi) = ( sqrt(psi) - sin(sqrt(psi)) ) / (sqrt(psi^3))
# for C2, (eta, H0) = (3,1/2)
# for C3, (eta, H0) = (5,1/6)


def stumpff_constraint(psi, H0, eta, eps, M):

    H = H0
    C = H0
    K = 2
    L = eta

    for i in range(M):
        H = -H * psi / (2 * K * L)
        if abs(H) <= eps:
            break
        C = C + H
        K = K + 1
        L = L + 2
    return C

# Algorithm 2: Compute C2 and C3 for any psi using Series and Recursion

# calculate C2 and C3 using series and recursion
# for my reference:
    # set K = 0
    # if abs(psi) <= psi_m:
    #     - compute C2 and C3 using stumpff_constraint: using algo 1 to get C2 & C3
    # K = K + 1 & psi = psi/4
    # keep checking the equality and keep solving for C2 & C3 until the equality is satisfied
    # if K = 0 --> end
    # set K = K-1 --> solve C3 and C2 using eqns
    # set psi = 4*psi and repeat until K = 0

def stumpff_C2_C3(psi, psi_m=1, eps=1e-12, M=50):

    K = 0

    while abs(psi) > psi_m:
        K = K + 1
        psi = psi/4

    (C2) = stumpff_constraint(psi, H0=1/2, eta=3, eps=eps, M=M)
    (C3) = stumpff_constraint(psi, H0=1/6, eta=5, eps=eps, M=M)

    # confused about the recursion part a little bit
    while K > 0:
        K = K - 1
        C3_new = (C2+C3-psi*C3*C2)/4
        C2_new = 1/2 * (1-psi*C3)**2
        psi = 4*psi
        C2 = C2_new
        C3 = C3_new
    return (C2, C3)


# """
# Verification functions using eqns

# got help from claude to write these, but they're just forumlas and basic math

def stumpff_C2_exact(psi):
    # Exact formula for C2(psi) for verification
    if psi == 0:
        return 0.5
    elif psi > 0:
        return (1 - np.cos(np.sqrt(psi))) / psi
    else:  # psi < 0
        return (np.cosh(np.sqrt(-psi)) - 1) / (-psi)


def stumpff_C3_exact(psi):
    # Exact formula for C3(psi) for verification
    if psi == 0:
        return 1/6
    elif psi > 0:
        return (np.sqrt(psi) - np.sin(np.sqrt(psi))) / (psi * np.sqrt(psi))
    else:  # psi < 0
        return (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / (-psi * np.sqrt(-psi))

# testing algorithm functions


# This was made by claude to test the stumpff functions
# ngl idk if its right bc no matter how u change psi_m or psi, it always passes.
'''
if __name__ == "__main__":
    print("Testing Stumpff Functions C2 and C3")
    print("=" * 60)
    
    # Test cases: various values of psi
    test_values = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, -0.1, -1.0, -5.0, 11.6446, -1.88569]
    
    print(f"{'psi':>10} {'C2 (algo)':>15} {'C2 (exact)':>15} {'Error':>12}")
    print(f"{'':>10} {'C3 (algo)':>15} {'C3 (exact)':>15} {'Error':>12}")
    print("-" * 60)
    
    for psi in test_values:
        C2_algo, C3_algo = stumpff_C2_C3(psi)
        C2_exact = stumpff_C2_exact(psi)
        C3_exact = stumpff_C3_exact(psi)
        
        error_C2 = abs(C2_algo - C2_exact)
        error_C3 = abs(C3_algo - C3_exact)
        
        print(f"{psi:10.5f} {C2_algo:15.10f} {C2_exact:15.10f} {error_C2:12.2e}")
        print(f"{'':10} {C3_algo:15.10f} {C3_exact:15.10f} {error_C3:12.2e}")
        print()
    
    print("\nTest passed! ✓" if all(
        abs(stumpff_C2_C3(psi)[0] - stumpff_C2_exact(psi)) < 1e-10 and
        abs(stumpff_C2_C3(psi)[1] - stumpff_C3_exact(psi)) < 1e-10
        for psi in test_values
    ) else "\nTest failed!")

'''
# Algorithm 3: Universal Lambert Solver

# clean up input r1 r2 TOF, run cases to assign PSI's 
# want to keep psi's the way they are irregardless of orbit type
# full out lambert to solve for a 
# first set up lamberts to solve for a,e,p then figure out way to mess with stumpff psi values, then fix to only input r1 r2 tof


# import poliastro and check with their function --> verify with their example 
# possibly work with izzo
#--------------------------------------------------------------------------------------------------------------------
def universal_lambert(r1_vec, r2_vec, TOF, mu,desired_path = 'short'):

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    
    # actual_angle = np.arccos(np.dot(r1_vec, r2_vec) / (r1 * r2))
    # print(f'Actual angle between position vectors: {np.rad2deg(actual_angle):.2f} deg')

    if desired_path == 'short':
        delta_f = np.arccos((np.dot(r1_vec, r2_vec)) / (r1*r2))
    elif desired_path == 'long':
        delta_f = (2*np.pi) - np.arccos((np.dot(r1_vec, r2_vec)) / (r1*r2))
    print(f'Delta F: {np.rad2deg(delta_f)} deg')
    
    c = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(delta_f)) 
    s = (r1 + r2 + c) / 2

    if 0 <= delta_f < np.pi:  
        t_parab = (1/3) * np.sqrt(2/mu) * (s**(3/2) - ((s-c)**(3/2)))
    elif np.pi <= delta_f < 2*np.pi:
        t_parab = (1/3) * np.sqrt(2/mu) * (s**(3/2) + ((s-c)**(3/2)))

    # this is where I can set the psi values
    if TOF > t_parab:
        psi_0 = 0.8
        psi_upper = 4 * math.pi**2
        psi_lower = -4 * math.pi**2
        # print("Transfer Orbit is Elliptical")
    elif TOF == t_parab:
        psi_0 = 0.8
        psi_upper = 4 * math.pi**2
        psi_lower = -4 * math.pi**2
        # print("Transfer Orbit is Parabolic")
    else:
        psi_0 = -0.1
        psi_upper = 4 * math.pi**2
        psi_lower = -2
        # print("Transfer Orbit is Hyperbolic")

#     
#     # Calculate minumum transfer
#     a_m = s/2
#     alpha_m = np.pi
#     if 0 <= delta_f < np.pi:  # type:ignore
#         beta_m = 2*np.arcsin(np.sqrt((s-c)/s))
#     elif np.pi <= delta_f < 2*np.pi:  # type:ignore
#         beta_m = -2*np.arcsin(np.sqrt((s-c)/s))

#     tm = np.sqrt((s**3)/(8 * mu)) * \
#         (np.pi - beta_m + np.sin(beta_m))  # type:ignore


#  # Solve for a, p and e
#     if TOF > t_parab: # elliptical case
#     # Define alpha and beta
#         if TOF <= tm:
#             def alpha(a): return 2*np.arcsin(np.sqrt((s/(2*a))))
#         elif TOF > tm:
#             def alpha(a): return 2*np.pi - 2*np.arcsin(np.sqrt((s/(2*a))))

#         if 0 <= delta_f < np.pi:  # type:ignore
#             def beta(a): return 2*np.arcsin(np.sqrt((s-c)/(2*a)))
#         elif np.pi <= delta_f < 2*np.pi:  # type:ignore
#             def beta(a): return -2*np.arcsin(np.sqrt((s-c)/(2*a)))

#         def lambert_eq(a): return ((np.sqrt(a**3)) * (alpha(a) - np.sin(alpha(a)) - beta(a) + np.sin(beta(a)))) - ((np.sqrt(mu))*TOF)

#         a = optimize.brentq(lambert_eq, a_m, a_m*100)
#         p = (((4*a)*(s-r1)*(s-r2))/(c**2)) * (np.sin((alpha(a) + beta(a))/2)**2)  
#         e = np.sqrt(1 - (p/a))

#     elif TOF < t_parab: # hyperbolic case

#         def alpha_h(a): return 2*np.arcsinh(np.sqrt(s/(-2*a)))
#         def beta_h(a): return 2*np.arcsinh(np.sqrt((s-c)/(-2*a)))

#         if 0 <= delta_f < np.pi:
#             def lambert_eq(a): return ((np.sqrt((-a)**3)) * (np.sinh(alpha_h(a)) - alpha_h(a) - np.sinh(beta_h(a)) + beta_h(a))) - (np.sqrt(mu)*TOF)
#         elif np.pi <= delta_f < 2*np.pi:
#             def lambert_eq(a): return ((np.sqrt((-a)**3)) * (np.sinh(alpha_h(a)) - alpha_h(a) + np.sinh(beta_h(a)) - beta_h(a))) - (np.sqrt(mu)*TOF)
        
#         a = optimize.brentq(lambert_eq, -a_m*1000, -a_m)
#         p = (((4*(-a))*(s-r1)*(s-r2))/(c**2)) * (np.sinh((alpha_h(a) + beta_h(a))/2)**2) 
#         e = np.sqrt(1 - (p/a))

    gamma = np.dot(r1_vec, r2_vec) / (r1 * r2)

    if desired_path == "short":
        t_m = 1
    elif desired_path == "long":
        t_m = -1

    A =  t_m*(r1 * r2 * (1 + gamma))**0.5
    if abs(A) < 0:
        # this line was suggested by VS code: I was gonna have a print statement here
        raise ValueError("Transfer angle is 180 degrees; Lambert's problem is undefined.")

    psi = psi_0

    # tolerance
    eps = 1e-12

    # Max iter
    M = 100

    for i in range(M):

        (C2, C3) = stumpff_C2_C3(psi, eps=eps, M=M)
        B = r1 + r2 + (1/np.sqrt(C2)) * (A * (psi * C3 - 1))
        # print(f'Iteration {i+1}: psi = {psi}, B = {B}')

        # paper says to readjust psi_lower until B > 0 if both A>0 and B<0

       # used if B < 0, otherwhise Chi would be complex
        ii = 0
        while B < 0 and ii < M:
            # used to move psi_lower up
            psi_lower = .5 * (psi + psi_lower)
            # in turn moves psi up
            psi = .5 * (psi_upper + psi_lower)
            C2, C3 = stumpff_C2_C3(psi, eps=eps, M=M)
            B = r1 + r2 + (1/np.sqrt(C2)) * (A * (psi * C3 - 1))
            ii += 1
            # print(psi_lower) gets hit if hyperbolic
            if ii == M:
                raise ValueError(
                    "Failed to find a positive B value within maximum iterations.")
            continue

        chi = np.sqrt(B/C2)
        delta_tt = (chi**3 * C3 + A * np.sqrt(B))/np.sqrt(mu)

        if abs(delta_tt - TOF) < eps:
            # compute F & G function eqns
            # end the for loop and return v1_vec and v2_vec
            break

        # step 7.1
        if delta_tt <= TOF:
            psi_lower = psi
        else:
            psi_upper = psi

        psi = .5 * (psi_upper + psi_lower)

    F = 1 - B/r1
    G = A * np.sqrt(B/mu)
    G_dot = 1 - B/r2

    v1_vec = 1/G * np.array([r2_vec[0] - F * r1_vec[0], r2_vec[1] - F * r1_vec[1], r2_vec[2] - F * r1_vec[2]])
    v2_vec = 1/G * np.array([G_dot * r2_vec[0] - r1_vec[0], G_dot * r2_vec[1] - r1_vec[1], G_dot * r2_vec[2] - r1_vec[2]])

    h = np.cross(r1_vec, v1_vec)
    e = np.linalg.norm((np.cross(v1_vec, h)/mu) - (r1_vec/r1))

    if e < 1-eps:
        orbit_type = "Elliptic"
    elif abs(e-1) > 10e-4:
        orbit_type = "Hyperbolic"
    elif abs(e-1) < 10e-4:
        orbit_type = "Parabolic"
    # print(f"Orbit is {orbit_type} with eccentricity {e:.10f}")

    # SMA = - mu / (2 * ( .5*np.linalg.norm(v2_vec**2 ) - (mu/r2)))
    a = - mu / (2 * ( .5*np.linalg.norm(v1_vec)**2 - (mu/r1)))
    p = a*(1-e**2)


    return a,p,e,v1_vec,v2_vec

    # return v1_vec, v2_vec, B, chi, psi, SMA
# --------------------------------------------------------------------------------------------------------------------


# # Elliptic Test Case
# v1_vec, v2_vec, B, chi, psi,SMA = universal_lambert(
#     np.array([1.01566, 0, 0]),
#     np.array([0.387926, 0.183961, 0.551884]),
#     TOF=5,
#     mu=1,
#     desired_path = 'short'
# )
# print(f'Final v1_vec = {v1_vec}')
# print(f'Final v2_vec = {v2_vec}')
# print(f'Final B = {B}, Final Chi = {chi}, Final Psi = {psi}\n')

# # Hyperbolic Test Case
# v1_vec, v2_vec, B, chi, psi, SMA = universal_lambert(
#     np.array([-0.668461, -2.05807, -1.9642]),
#     np.array([3.18254, 2.08111, -4.89447]),
#     TOF=5,
#     mu=1
# )
# print(f'Final v1_vec = {v1_vec}')
# print(f'Final v2_vec = {v2_vec}')
# print(f'Final B = {B}, Final Chi = {chi}, Final Psi = {psi}\n')

# # Parabolic Test Case
# v1_vec, v2_vec, B, chi, psi, SMA = universal_lambert(
#     np.array([-0.253513, 1.21614, -1.20916]),
#     np.array([-0.434366, 4.92818, 0.0675545]),
#     TOF=5,
#     mu=1
# )
# print(f'Final v1_vec = {v1_vec}')
# print(f'Final v2_vec = {v2_vec}')
# print(f'Final B = {B}, Final Chi = {chi}, Final Psi = {psi}\n')