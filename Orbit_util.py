from orbit import *
import numpy as np
from scipy.integrate import ode
from scipy import optimize
import matplotlib.pyplot as plt
from body import Body
from astropy import constants as const


G = 6.674 * 10**-11  # m^3 kg^-1 s^-2


def lambert_solver(r1_vec, r2_vec, tof, mu, desired_path='short'):
    """
    Returns a, e, p or the transfer orbit
    Follows 458 lectures
    Uses method described in Prussing, John E., and Bruce A. Conway. Orbital Mechanics. Oxford University Press, 2013. 
    """

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    if desired_path == 'short':
        delta_f = np.arccos((np.dot(r1_vec, r2_vec)) / (r1*r2))
    elif desired_path == 'long':
        delta_f = (2*np.pi) - np.arccos((np.dot(r1_vec, r2_vec)) / (r1*r2))
    print(f'Delta F: {np.rad2deg(delta_f)} deg')

    # Calc chord and space triangel perimeter
    c = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(delta_f))  # type:ignore
    s = (r1 + r2 + c) / 2

    # Calc t_parab
    if 0 <= delta_f < np.pi:  # type:ignore
        t_parab = (1/3) * np.sqrt(2/mu) * (s**(3/2) - ((s-c)**(3/2)))
    elif np.pi <= delta_f < 2*np.pi:  # type:ignore
        t_parab = (1/3) * np.sqrt(2/mu) * (s**(3/2) + ((s-c)**(3/2)))

    if tof > t_parab:  # type:ignore
        # print("elliptical solution")
        pass
    else:
        print('hyperbolic solution')
        print(f"{r1_vec}")
        print(f"{r2_vec}")

        # ValueError('Hyperbolic Solution')
        return None

    # Calculate minumum transfer
    a_m = s/2
    alpha_m = np.pi
    if 0 <= delta_f < np.pi:  # type:ignore
        beta_m = 2*np.arcsin(np.sqrt((s-c)/s))
    elif np.pi <= delta_f < 2*np.pi:  # type:ignore
        beta_m = -2*np.arcsin(np.sqrt((s-c)/s))

    tm = np.sqrt((s**3)/(8 * mu)) * \
        (np.pi - beta_m + np.sin(beta_m))  # type:ignore

    # Define alpha and beta
    if tof <= tm:
        def alpha(a): return 2*np.arcsin(np.sqrt((s/(2*a))))
    elif tof > tm:
        def alpha(a): return 2*np.pi - 2*np.arcsin(np.sqrt((s/(2*a))))

    if 0 <= delta_f < np.pi:  # type:ignore
        def beta(a): return 2*np.arcsin(np.sqrt((s-c)/(2*a)))
    elif np.pi <= delta_f < 2*np.pi:  # type:ignore
        def beta(a): return -2*np.arcsin(np.sqrt((s-c)/(2*a)))

    # Solve for a, p and e
    def lambert_eq(a): return ((np.sqrt(a**3)) * (alpha(a) - np.sin(alpha(a)
                                                                    ) - beta(a) + np.sin(beta(a)))) - ((np.sqrt(mu))*tof)
    a = optimize.brentq(lambert_eq, a_m, a_m*100)
    p = (((4*a)*(s-r1)*(s-r2))/(c**2)) * \
        (np.sin((alpha(a) + beta(a))/2)**2)  # type:ignore
    e = np.sqrt(1 - (p/a))

    # Calculate v1 @ r1 and v2 @ r2
    # Calc unit vectors
    u1 = r1_vec / r1
    u2 = r2_vec / r2
    uc = (r2_vec - r1_vec) / c

    A = np.sqrt(mu/(4*a))*(1/np.tan(alpha(a)/2))  # type:ignore
    B = np.sqrt(mu/(4*a))*(1/np.tan(beta(a)/2))  # type:ignore

    v1 = (B+A)*uc + (B-A)*u1
    v2 = (B+A)*uc - (B-A)*u2

    return a, p, e, v1, v2


def propogate_orbit(r, v, mu, tspan, dt):
    """
    Propgates orbit based on given r and v vectores. 
    Returns series of position vectors 'rs'
    Returns series of velocity vectors 'vs'
    """
    n_steps = int(np.ceil(tspan/dt))
    ys = np.zeros((n_steps, 6))
    ts = np.zeros((n_steps, 1))

    # Intial condition of solver
    y0 = np.concatenate((r, v))
    ys[0] = y0
    step = 1

    # Intiate Solver
    solver = ode(y_dot)
    solver.set_integrator('lsoda')
    solver.set_initial_value(y0, 0)
    solver.set_f_params(mu)

    while solver.successful() and step < n_steps:
        solver.integrate(solver.t+dt)
        ts[step] = solver.t
        ys[step] = solver.y
        step += 1

    rs = ys[:, :3]
    vs = ys[:, 3:6]

    return (rs, vs)


def orb_2_pqw(r, f, e, p, mu, degrees=True):
    """
    Transforms orbital frame to perifocal frame
    """
    if degrees:
        f = np.deg2rad(f)
    else:
        pass
    r_pqw = np.array([r*np.cos(f), r*np.sin(f), 0])
    v_pqw = np.array(
        [-np.sqrt(mu/p)*np.sin(f), np.sqrt(mu/p)*(e + np.cos(f)), 0])

    return r_pqw, v_pqw


def perif_2_eci(r_pqw, v_pqw, inc, raan, aop):
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
    r_eci = perif_2_eci_DCM @ r_pqw
    v_eci = perif_2_eci_DCM @ v_pqw

    return r_eci, v_eci


def y_dot(t, y, mu):
    """
    Two Body physics used for propagation
    """
    # print(y)
    rx, ry, rz, vx, vy, vz = y  # Deconstruct State to get r_vec
    r = np.array([rx, ry, rz])
    r_norm = np.linalg.norm(r)
    ax, ay, az = -r*mu/r_norm**3  # Two body Problem ODE
    return [vx, vy, vz, ax, ay, az]


def y_dot_n_body(t, y, central_body: Body, n_bodies: int, bodies: list[Body]):
    """
    N Body physics used for propagation cause im goated
    """
    y_dot = np.zeros(n_bodies*6)

    i = 0
    for _ in range(len(bodies)):

        y_nbody = y[i:i+6]
        r_nbody = y_nbody[0:3]
        v_nbody = y_nbody[3:6]
        r_nbody_mag = np.linalg.norm(r_nbody)
        a_nbody = ((-G*central_body.mass)/(r_nbody_mag**3))*r_nbody

        ii = 0
        for body in bodies:
            y_kbody = y[ii:ii+6]

            if (y_kbody == y_nbody).all():
                pass
            else:
                r_kbody = y_kbody[0:3]
                v_kbody = y_kbody[3:6]
                r_kbody_mag = np.linalg.norm(r_kbody)

                r = r_kbody - r_nbody
                r_mag = np.linalg.norm(r)
                a_nbody += ((G*body.mass)/(r_mag**3)) * r

            ii += 6

        y_dot[i:i+6] = np.concatenate((v_nbody, a_nbody))
        i += 6

    return y_dot
