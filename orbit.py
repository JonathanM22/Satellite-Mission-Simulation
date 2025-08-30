import numpy as np
from scipy.integrate import ode

# Define variables to be imported
# when module is imported
__all__ = ['Orbit', 'orb_2_pqw', 'perif_2_eci', 'propogate_orbit']


class Orbit:
    """
    Intializes orbit class with 6 orbital elements. 
    """

    def __init__(self, mu, a=None, e=None, f0=None, inc=None, raan=None, aop=None, degrees=True):
        self.a = a
        self.e = e
        self.inc = np.deg2rad(inc) if (inc is not None and degrees) else inc
        self.raan = np.deg2rad(raan) if (
            raan is not None and degrees) else raan
        self.aop = np.deg2rad(aop) if (aop is not None and degrees) else aop
        self.mu = mu
        self.h = None
        self.p = None

    def r_at_true_anomaly(self, f, degrees=True):
        """
        Returns r scalor using orbit equation 
        """
        try:
            if degrees:
                f = np.deg2rad(f)
            return self.calc_p() / (1 + self.e*np.cos(f))
        except:
            raise ValueError(
                "r_at_true_anomaly cannot be calculated. Check a, e or f")

    def energy(self):
        """
        Returns orbit energy 
        """
        try:
            return self.a * (1 - self.e**2)  # type:ignore
        except:
            raise ValueError(
                "Orbit energy  cannot be calculated. Check a")

    def calc_p(self):
        """
        Returns semilatus rectum  
        """
        if self.p is not None:
            return self.p
        else:
            try:
                return self.a * (1 - self.e**2)  # type:ignore
            except:
                raise ValueError(
                    "Orbit semilatus rectum cannot be calculated. Check a or e")

    def period(self):
        """
        Calculates period or orbit.
        Returns period in seconds
        """
        try:
            return (2*np.pi)*np.sqrt(self.a**3/self.mu)  # type:ignore
        except:
            raise ValueError(
                "Orbit has period cannot be calculated. Check a or mu")


def propogate_orbit(self, r, v, tspan, dt):
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
    solver.set_f_params(self.mu)

    while solver.successful and step < n_steps:
        solver.integrate(solver.t+dt)
        ts[step] = solver.t
        ys[step] = solver.y
        step += 1

    rs = ys[:, :3]
    vs = ys[:, 3:]

    return (rs, vs)


def orb_2_pqw(r, f, e, p, mu):
    """
    Transforms orbital frame to perifocal frame
    """
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
