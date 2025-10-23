import numpy as np
import math

def main():
    # Vraj Patel
    # 993932969
    # I have completed this with integrity

    mu = 4 * math.pi**2

    # Case 1
    r01 = np.array([1.5321, -3.0259, 0.4806])
    v01 = np.array([1.8167, 2.2898, -1.4727])
    t01 = 0
    t11 = 1.450
    print("Case 1:")
    e1, a1, energy10, h10, sig01, alpha01 = orbitparm(r01, v01)
    chi1, w1, z1, q1, U_case1 = NR_uni_time_eqn(t11, t01, sig01, alpha01, np.linalg.norm(r01), mu)
    r11, v11 = fgfun(alpha01, w1, z1, r01, v01, mu, sig01, t11)
    energy11, h11 = EnergyAngMom(r11, v11, h10, energy10, t11)
    testcase(U_case1, alpha01, chi1)

    # Case 2
    r02 = np.array([1.4133, -0.2948, 0.2448])
    v02 = np.array([3.9221, 4.7680, -5.5161])
    t02 = 0
    t12 = 1.853
    print("\n\nCase 2:")
    e2, a2, energy20, h20, sig02, alpha02 = orbitparm(r02, v02)
    chi2, w2, z2, q2, U_case2 = NR_uni_time_eqn(t12, t02, sig02, alpha02, np.linalg.norm(r02), mu)
    r12, v12 = fgfun(alpha02, w2, z2, r02, v02, mu, sig02, t12)
    energy21, h21 = EnergyAngMom(r12, v12, h20, energy20, t12)
    testcase(U_case2, alpha02, chi2)

# Functions

def orbitparm(r0, v0):
    """Function for orbit, eccentricity, semi major axis, energy, ang mom"""
    mu = 4 * math.pi**2
    r0_norm = np.linalg.norm(r0)
    v0_norm = np.linalg.norm(v0)
    
    energy = 0.5 * v0_norm**2 - mu / r0_norm
    
    if energy > 0:
        print("Orbit is Hyperbolic")
    elif energy == 0:
        print("Orbit is Parabolic")
    else:
        print("Orbit is Elliptical")
    
    h = np.cross(r0, v0)
    cross_term = np.cross(v0, h)
    e_vector = (1/mu) * cross_term - (r0/r0_norm)
    e = np.linalg.norm(e_vector)
    
    print(f"\nEccentricity is {e:.4f}")
    a = -mu/(2*energy)
    print(f"\nSemi-Major axis is {a:.4f} LU")
    sig = np.dot(r0, v0) / math.sqrt(mu)
    alpha = 1/a
    
    return e, a, energy, h, sig, alpha

def topdownW(chi, alpha):
    """Top down for solving w"""
    Delta = 1e-6
    
    # Initialize first values
    a_val = 0.5 * chi
    b_val = 1
    delta = 1
    sigma = a_val / b_val
    W = sigma
    n = 1
    while abs(W) >= Delta:
        n += 1
        a_new = alpha * (a_val)**2
        b_new = 2*n - 1
        delta = 1 / (1 - (a_new / (b_new * b_val)) * delta)
        W = W * (delta - 1)
        sigma += W
        b_val = b_new
    return sigma

def topdownz(q):
    """Top down for z"""
    Delta = 1e-6
    
    # Initialize first values
    a_val = 1
    b_val = 1
    delta = 1
    sigma = a_val / b_val
    W = sigma
    n = 1
    while abs(W) >= Delta:
        n += 1
        # Calculate gamma based on even/odd n
        if n % 2 == 0:  # even n
            gamma = ((n-1) + 2) * ((n-1) + 5) / ((2*(n-1) + 1) * (2*(n-1) + 3))
        else:  # odd n
            gamma = (n-1) * ((n-1) - 3) / ((2*(n-1) + 1) * (2*(n-1) + 3))
        
        a_new = q * gamma
        delta = 1 / (1 - (a_new / b_val) * delta)
        W = W * (delta - 1)
        sigma += W
    return sigma

def uniconsts(alpha, w, z):
    """Function for u0,u1,u2,u3"""
    U0 = (1 - alpha * w**2) / (1 + alpha * w**2)
    U1 = (2 * w) / (1 + alpha * w**2)
    U2 = (2 * w**2) / (1 + alpha * w**2)
    U3 = 2/3 * (U1**3) * z - U1 * U2
    
    return U0, U1, U2, U3

def NR_uni_time_eqn(t, t0, sig0, alpha, r0, mu):
    """Function to solve Chi using newton raphson of universal time eq"""
    Delta = 1e-6
    chi = 1.0  # initial guess
    diff = float('inf')
    
    while abs(diff) >= Delta:
        W = topdownW(chi, alpha)
        Q = (alpha * W**2) / (1 + alpha * W**2)
        Z = topdownz(Q)
        U0, U1, U2, U3 = uniconsts(alpha, W, Z)
        
        f = r0 * U1 + sig0 * U2 + U3 - math.sqrt(mu) * (t - t0)
        f_prime = r0 * U0 + sig0 * U1 + U2
        
        chi_new = chi - f / f_prime
        diff = chi_new - chi
        chi = chi_new
    
    print(f'\nFinal chi value: {chi:.6f}')
    print(f'w: {W:.6f}')
    print(f'z: {Z:.6f}')
    print(f'q: {Q:.6f}')
    
    U = [U0, U1, U2, U3]
    return chi, W, Z, Q, U

def fgfun(alpha, w, z, r0, v0, mu, sig, t1):
    """Function for finding r1 and v1"""
    _, U1, U2, _ = uniconsts(alpha, w, z)
    
    r0_norm = np.linalg.norm(r0)
    
    F = 1 - (1/r0_norm) * U2
    G = (r0_norm/math.sqrt(mu)) * U1 + (sig/math.sqrt(mu)) * U2
    r1 = F * r0 + G * v0
    
    r1_norm = np.linalg.norm(r1)
    Ft = -math.sqrt(mu) * U1 / (r0_norm * r1_norm)
    Gt = 1 - (1/r1_norm) * U2
    v1 = Ft * r0 + Gt * v0
    
    print(f'At time {t1:.3f} TU the position vector r1 is [{r1[0]:.4f}, {r1[1]:.4f}, {r1[2]:.4f}]')
    print(f'\nAt time {t1:.3f} TU the velocity vector v1 is [{v1[0]:.4f}, {v1[1]:.4f}, {v1[2]:.4f}]\n')
    
    return r1, v1

def EnergyAngMom(r1, v1, h0, energy0, t1):
    """Energy and angular momentum conservation check"""
    mu = 4 * math.pi**2
    r1_norm = np.linalg.norm(r1)
    v1_norm = np.linalg.norm(v1)
    
    energy = 0.5 * v1_norm**2 - mu / r1_norm
    h = np.cross(r1, v1)
    
    if abs(energy0 - energy) <= 1e-6:
        print(f"Energy is conserved: at t = {t1:.3f}TU, ε0 = {energy:.4f}, at t = 0, ε1 = {energy0:.4f}")
    
    h0_norm = np.linalg.norm(h0)
    h_norm = np.linalg.norm(h)
    if abs(h0_norm - h_norm) <= 1e-6:
        print(f"\nAng Mom is conserved: at t = {t1:.3f}TU, h = [{h[0]:.4f}, {h[1]:.4f}, {h[2]:.4f}], at t = 0, h0 = [{h0[0]:.4f}, {h0[1]:.4f}, {h0[2]:.4f}]")
    
    return energy, h

def testcase(U, alpha, chi):
    """Test case function"""
    if alpha > 0:  # ellipse
        sqrt_alpha = math.sqrt(alpha)
        cos_term = math.cos(sqrt_alpha * chi)
        sin_term = math.sin(sqrt_alpha * chi)
        
        U0_known = cos_term
        U2_known = (1 - cos_term) / alpha
        U3_known = (sqrt_alpha * chi - sin_term) / (alpha * sqrt_alpha)
        
        if abs(U0_known - U[0]) <= 1e-6:
            print(f' \nNumerically calculated U0 matches with known value = {U[0]:.4f}')
        if abs(U2_known - U[2]) <= 1e-6:
            print(f' \nNumerically calculated U2 matches with known value = {U[2]:.4f}')
        if abs(U3_known - U[3]) <= 1e-6:
            print(f' \nNumerically calculated U3 matches with known value = {U[3]:.4f}')
    else:  # hyperbola
        sqrt_neg_alpha = math.sqrt(-alpha)
        U1_known = math.sinh(sqrt_neg_alpha * chi) / sqrt_neg_alpha
        
        if abs(U1_known - U[1]) <= 1e-6:
            print(f' \nNumerically calculated U1 matches with known value = {U[1]:.4f}')
    
    return U[0], U[1], U[2], U[3]

if __name__ == "__main__":
    main()