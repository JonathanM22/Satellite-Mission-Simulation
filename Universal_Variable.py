import numpy as np
import math

# Algorithm 1: Computation of Stumpff Functions C2 and C3
    # set C2(psi) = ( 1 - cos(sqrt(psi)) ) / psi) 
    # set C3(psi) = ( sqrt(psi) - sin(sqrt(psi)) ) / (sqrt(psi^3))
# for C2, (eta, H0) = (3,1/2) 
# for C3, (eta, H0) = (5,1/6)

def stumpff_constraint(psi,H0,eta,eps=1e-8,M=100):

    H = H0
    C = H0
    K = 2 
    L = eta

    for i in range(M):
        H = -H * psi / (2 *K  * L)
        if abs(H) <= eps:
            break
        N = i
        C = C + H
        K = K + 1
        L = L + 2
        
    return C, N


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

def stumpff_C2_C3(psi,psi_m):
    
    K = 0 

    while abs(psi) <= psi_m:
        C2, N2 = stumpff_constraint(psi,1/2,3)
        C3, N3 = stumpff_constraint(psi,1/6,5)
        K = K + 1
        psi = psi/4
    return C2, C3, N2, N3






# Algorithm 3: Universal Lambert Solver