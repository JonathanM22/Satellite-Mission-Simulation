import numpy as np
import math

# Algorithm 1: Computation of Stumpff Functions C2 and C3
    # set C2(psi) = ( 1 - cos(sqrt(psi)) ) / psi) 
    # set C3(psi) = ( sqrt(psi) - sin(sqrt(psi)) ) / (sqrt(psi^3))
# for C2, (eta, H0) = (3,1/2) 
# for C3, (eta, H0) = (5,1/6)

def stumpff_constraint(psi,H0,eta,eps,M):

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

def stumpff_C2_C3(psi,psi_m=1,eps=1e-12,M=50):
    
    K = 0 

    while abs(psi) > psi_m:
        K = K + 1
        psi = psi/4

    (C2) = stumpff_constraint(psi,H0=1/2,eta=3,eps=eps,M=M)
    (C3) = stumpff_constraint(psi,H0=1/6,eta=5,eps=eps,M=M)

    # confused about the recursion part a little bit
    while K > 0: 
        K = K - 1
        C3_new = (C2+C3-psi*C3*C2)/4
        C2_new = 1/2 * (1-psi*C3)**2
        psi = 4*psi
        C2 = C2_new
        C3 = C3_new
    return (C2,C3)


#"""
# Verification functions using eqns

# got help from claude to write these, but they're just forumlas and basic math

def stumpff_C2_exact(psi):
    #Exact formula for C2(psi) for verification
    if psi == 0:
        return 0.5
    elif psi > 0:
        return (1 - np.cos(np.sqrt(psi))) / psi
    else:  # psi < 0
        return (np.cosh(np.sqrt(-psi)) - 1) / (-psi)

def stumpff_C3_exact(psi):
    #Exact formula for C3(psi) for verification
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
def universal_lambert(r1_vec, r2_vec, TOF, psi_0, psi_upper,psi_lower,M, eps,tm,mu):
    
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    gamma = np.dot(r1_vec, r2_vec) / (r1 * r2)
    beta = tm * (1-gamma**2)**0.5
    A = tm * (r1 * r2 * (1 + gamma))**0.5

    if abs(A) < 0:
        # this line was suggested by VS code: I was gonna have a print statement here
        raise ValueError("Transfer angle is 180 degrees; Lambert's problem is undefined.")
    
    psi = psi_0

    for i in range(M):

        (C2,C3) = stumpff_C2_C3(psi,eps=eps,M=M)
        B = r1 + r2 + (1/math.sqrt(C2)) * (A * (psi * C3 - 1))
        # print(f'Iteration {i+1}: psi = {psi}, B = {B}')

        # paper says to readjust psi_lower until B > 0 if both A>0 and B<0

       # used if B < 0, otherwhise Chi would be complex
        i = 0
        while B < 0 and i < M:
            # used to move psi_lower up
            psi_lower = .5 * (psi + psi_lower)
            # in turn moves psi up
            psi = .5 * (psi_upper + psi_lower)
            C2, C3 = stumpff_C2_C3(psi, eps=eps, M=M)
            B = r1 + r2 + (1/math.sqrt(C2)) * (A * (psi * C2 - 1))
            i += 1
            if i == M:
                raise ValueError("Failed to find a positive B value within maximum iterations.")
            continue

        chi = math.sqrt(B/C2)
        delta_tt = (chi**3 * C3 + A * math.sqrt(B))/math.sqrt(mu)

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

        #   psi_lower = psi``
        #   psi_current = .5 * (psi_upper + psi_lower)    
        #   psi_0 = psi_current
        #   continue
        # psi_upper = psi

    F = 1 - B/r1
    G = A * math.sqrt(B/mu)
    G_dot = 1 - B/r2

    v1_vec = 1/G * np.array([r2_vec[0] - F * r1_vec[0], r2_vec[1] - F * r1_vec[1], r2_vec[2] - F * r1_vec[2]])
    v2_vec = 1/G * np.array([G_dot * r2_vec[0] - r1_vec[0], G_dot * r2_vec[1] - r1_vec[1], G_dot * r2_vec[2] - r1_vec[2]])

    h = np.cross(r1_vec, v1_vec)
    e = np.linalg.norm( np.cross(v1_vec,h)/mu - r1_vec/r1)

    if e < 1:
        orbit_type = "Elliptic"
    elif e > 1:
        orbit_type = "Hyperbolic"
    else:
        orbit_type = "Parabolic"
    print(f"\nOrbit is {orbit_type} with eccentricity {e:.10f}")
    

    return v1_vec, v2_vec, B, chi, psi

# Elliptic Test Case
v1_vec, v2_vec, B, chi, psi = universal_lambert(
    np.array([1.01566, 0, 0]),
    np.array([0.387926, 0.183961, 0.551884]),
    TOF = 5,
    psi_0 = 0.8,
    psi_upper = 4 * math.pi**2,
    psi_lower = -4 * math.pi**2,
    M = 50, 
    eps = 1e-10,
    tm = 1,
    mu = 1
)
print(f'Final v1_vec = {v1_vec}')
print(f'Final v2_vec = {v2_vec}')
print(f'Final B = {B}, Final Chi = {chi}, Final Psi = {psi}\n')


# Parabolic Test Case
v1_vec, v2_vec, B, chi, psi = universal_lambert(
    np.array([-0.253513, 1.21614, -1.20916]),
    np.array([-0.434366, 4.92818, 0.0675545]),
    TOF = 5,
    psi_0 = 0.8,
    psi_upper = 4 * math.pi**2,
    psi_lower = -4 * math.pi**2,
    M = 50, 
    eps = 1e-6,
    tm = 1,
    mu = 1
)
print(f'Final v1_vec = {v1_vec}')
print(f'Final v2_vec = {v2_vec}')
print(f'Final B = {B}, Final Chi = {chi}, Final Psi = {psi}\n')


# Hyperbolic Test Case
v1_vec, v2_vec, B, chi, psi = universal_lambert(
    np.array([-0.668461, -2.05807, -1.9642 ]),
    np.array([3.18254, 2.08111, -4.89447 ]),
    TOF = 5,
    psi_0 = -0.1,
    psi_upper = 4 * math.pi**2,
    psi_lower = -2,
    M = 50, 
    eps = 1e-7,
    tm = 1,
    mu = 1
)
print(f'Final v1_vec = {v1_vec}')
print(f'Final v2_vec = {v2_vec}')
print(f'Final B = {B}, Final Chi = {chi}, Final Psi = {psi}\n')

