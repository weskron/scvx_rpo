import numpy as np
import math

class KeplerianElements:
    def __init__(self, semi_major_axis: float, eccentricity: float, inclination: float, raan: float, argp: float, nu: float = None, mean_anomaly: float = None):
        self.semi_major_axis = semi_major_axis   # m
        self.eccentricity = eccentricity       # eccentricity
        self.inclination = inclination        # rad
        self.raan = raan     # right ascension of ascending node, rad
        self.argp = argp     # argument of periapsis, rad
        self.nu = nu if nu is not None else mean_to_true_anomaly(mean_anomaly, eccentricity)   # true anomaly, rad
        self.mean_anomaly = mean_anomaly if mean_anomaly is not None else true_to_mean_anomaly(nu, eccentricity) # Mean anomaly, rad
        self.validate()

    def validate(self):
        if self.semi_major_axis < 0:
            raise ValueError(f"Semi major axis must be positive, entry was {self.semi_major_axis}")

        if self.inclination < 0 or self.inclination > 2*np.pi:
            raise ValueError(f"Inclination must be between 0 and 2 pi, entry was {self.inclination}")

        if self.raan < 0 or self.raan >= 2*np.pi:
            raise ValueError(f"Right ascension of the ascending node must be between 0 and 2 pi, entry was {self.raan}")

        if self.argp < 0 or self.argp > 2*np.pi:
            raise ValueError(f"Argument of periapsis must be between 0 and 2 pi, entry was {self.argp}")
        
        if self.nu < 0 or self.nu > 2*np.pi:
            raise ValueError(f"True Anomaly must be between 0 and 2 pi, entry was {self.argp}")
            
        if self.mean_anomaly < 0 or self.mean_anomaly > 2*np.pi:
            raise ValueError(f"Mean Motion must be between 0 and 2 pi, entry was {self.argp}")

def true_to_mean_anomaly(nu, eccentricity):
    if nu is None:
        raise ValueError("Please enter true anomaly value")
    eccentric_anomaly = true_to_eccentric(nu, eccentricity)
    mean_anomaly = kepler_eqn(eccentric_anomaly, eccentricity)
    return mean_anomaly

def mean_to_true_anomaly(mean_anomaly, eccentricity):
    if mean_anomaly is None:
        raise ValueError("Please enter mean anomaly value")
    eccentric_anomlay = solve_kepler(mean_anomaly, eccentricity)
    nu = eccentric_to_true(eccentric_anomlay, eccentricity)
    return nu

def solve_kepler(M, e, tol=1e-10, max_iter=50):
    if e < 1:  # Elliptic
        M = np.mod(M, 2*np.pi)

        E = M if e < 0.8 else np.pi

        for _ in range(max_iter):
            f = E - e*np.sin(E) - M
            fp = 1 - e*np.cos(E)

            if abs(fp) < 1e-12:
                break

            dE = -f / fp
            E += dE

            if abs(dE) < tol:
                return E

        return E

    else:  # Hyperbolic
        # Initial guess
        H = np.log(2*M/e + 1.8) if M > 0 else -np.log(-2*M/e + 1.8)

        for _ in range(max_iter):
            f = e*np.sinh(H) - H - M
            fp = e*np.cosh(H) - 1

            if abs(fp) < 1e-12:
                break

            dH = -f / fp
            H += dH

            if abs(dH) < tol:
                return H

        return H

def anomaly_to_true(anom, e):
    """
    Convert Eccentric/Hyperbolic anomaly to true anomaly
    """
    if anom is None:
        raise ValueError("Need entry for mean anomaly")
    
    if e < 1:
        # Elliptic
        E = anom
        nu = 2 * np.arctan2(
            np.sqrt(1+e)*np.sin(E/2),
            np.sqrt(1-e)*np.cos(E/2)
        )
    else:
        # Hyperbolic
        H = anom
        nu = 2 * np.arctan2(
            np.sqrt(e+1)*np.sinh(H/2),
            np.sqrt(e-1)*np.cosh(H/2)
        )

    return np.mod(nu, 2*np.pi)

def kepler_eqn(E, e):
    """
    Classic Kepler Equation
    
    Parameters
    ----------
    E : float
        Eccentric Anomaly [rad]
    e : float
        Eccentrity

    Returns
    -------
    M : float
        Mean Anomaly [rad]
    """
    M = E - e * np.sin(E)
    M = np.mod(M, 2 * np.pi)
    return M

def eccentric_to_true(E, e):
    """
    Convert Eccentric Anomaly to True Anomaly (nu). 

    Parameters
    ----------
    E : float
        Eccentric anomaly [rad]
    e : float
        Eccentricity 0 < e < 1. 

    Returns
    -------
    nu : float
        True anomaly [rad]
    """
    nu = 2 * math.atan2(math.sqrt(1+e)*math.sin(E/2), math.sqrt(1-e) * math.cos(E/2))
    nu = np.mod(nu, 2*np.pi)
    return nu

def true_to_eccentric(nu, e):
    """
    Convert True Anomaly (nu) to Eccentric Anomaly (E). 

    Parameters
    ----------
    nu : float
        True anomaly [rad]
    e : float
        Eccentricity 0 < e < 1. 

    Returns
    -------
    E : float
        Eccentric anomaly [rad]
    """
    E = 2 * np.arctan2(np.sqrt(1-e) * np.sin(nu/2), np.sqrt(1+e) * np.cos(nu/2))
    E = np.mod(E, 2*np.pi)
    return E

def keplerian_to_eci(keplerian: KeplerianElements, mu):
    """
    Convert Keplerian orbital elements to position/velocity in the
    inertial frame. Elliptic orbits only (e < 1).
    """
    e = keplerian.eccentricity
    nu = keplerian.nu
    a = keplerian.semi_major_axis

    # Eccentric anomaly, closed-form from true anomaly (no iteration needed
    # since KeplerianElements already stores a self-consistent nu)
    E = true_to_eccentric(nu, e)

    # Perifocal-frame radius, position, velocity
    r_c = a * (1 - e * np.cos(E))
    h = np.sqrt(mu * a * (1 - e**2))

    pos_o = r_c * np.array([np.cos(nu), np.sin(nu), 0.0])
    vel_o = (mu / h) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # Rotate perifocal -> inertial frame
    dcm = RTN_to_inert_313(keplerian.raan, keplerian.inclination, keplerian.argp)
    pos_i = dcm @ pos_o
    vel_i = dcm @ vel_o

    return pos_i, vel_i

def eci_to_keplerian(r_vec, v_vec, mu):
    """
    Convert position/velocity in the inertial frame to Keplerian orbital
    elements. Elliptic orbits only (0 <= e < 1); does not special-case
    circular or equatorial orbits (RAAN/argp are undefined in those cases).
    """
    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)
    if h_norm < 1e-10:
        raise ValueError("r_vec and v_vec are parallel; orbit has zero angular momentum (not a valid closed orbit)")

    r_hat = r_vec / np.linalg.norm(r_vec)
    e_vec = (np.cross(v_vec, h_vec) / mu) - r_hat
    e = np.linalg.norm(e_vec)

    k_vec = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)
    if n < 1e-10:
        raise ValueError("Orbit is equatorial (inclination ~0); RAAN is undefined")

    cos_nu = np.dot(e_vec, r_vec) / (e * np.linalg.norm(r_vec))
    nu = math.acos(np.clip(cos_nu, -1.0, 1.0))
    if np.dot(r_vec, v_vec) < 0:
        nu = 2 * np.pi - nu

    E = true_to_eccentric(nu, e)
    M = kepler_eqn(E, e)

    inc = np.arccos(np.clip(h_vec[2] / np.linalg.norm(h_vec), -1.0, 1.0))

    raan = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
    if n_vec[1] < 0:
        raan = 2 * np.pi - raan

    cos_aop = np.dot(n_vec, e_vec) / (n * np.linalg.norm(e_vec))
    aop = np.arccos(np.clip(cos_aop, -1.0, 1.0))
    if e_vec[2] < 0:
        aop = 2 * np.pi - aop

    a = 1 / ((2 / np.linalg.norm(r_vec)) - (np.linalg.norm(v_vec)**2 / mu))

    return KeplerianElements(
        semi_major_axis=a,
        eccentricity=e,
        inclination=inc,
        raan=raan,
        argp=aop,
        mean_anomaly=M,   # nu gets derived automatically in the constructor
    )

def RTN_to_inert_313(RAAN, inc, aop):
    """
    Convert 313 Euler angles (RAAN, inc, aop) to rotation matrix. Rotates orbital frame to intertial frame. 

    Parameters:
    RAN, inc, aop : float
        Euler angles in radians
        RAAN : RAAN, first rotation about z-axis
        inc: Inclination, rotation about x-axis
        aop: AOP, second rotation about z-axis

    Returns:
    R : 3x3 numpy array
        Rotation matrix
    """
    cO = np.cos(RAAN)
    sO = np.sin(RAAN)
    ci = np.cos(inc)
    si = np.sin(inc)
    cw = np.cos(aop)
    sw = np.sin(aop)

    R = np.array([
        [cO*cw - sO*ci*sw, -cO*sw - sO*ci*cw,  sO*si],
        [sO*cw + cO*ci*sw, -sO*sw + cO*ci*cw, -cO*si],
        [si*sw,             si*cw,             ci]
    ])
    return R


def inert_to_RTN_313(RAAN, inc, aop):
    R = RTN_to_inert_313(RAAN, inc, aop).T
    return R