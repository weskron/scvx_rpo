import numpy as np
import math
from proxops.kepler import *

import numpy as np

def keplerian_to_cartesian(a , e, inc, RAAN, aop, anomaly, mu, frame="inert", anomaly_type="mean"):
    """
    Convert from Keplerian orbital elements to cartesian coordinates in the inertial frame. 

    Parameters:
    a : float
        semi-major axis [km]
    e : float
        eccentricity 
    inc : float
        inclination [rad]
    RAAN : float
        Right Ascension of the Ascending Node [rad]
    aop : float
        Argument of Periapsis [rad]
    M : float
        Mean Anomaly [rad]
    mu : float
        Gravitational parameter [km^3/s^2]
    anomaly_type: str
    frame : str

    Output: 
    x : float
        x position component
    y : float
        y position component
    z : float
        z position component
    xdot : float
        x velocity component
    ydot : float
        y velocity component
    zdot : float
        z velocity component
    """
    # Step 1: Find True Anomaly (nu) and Eccentric Anomaly (E)
    if anomaly_type == "mean":
        E = solve_kepler(anomaly, e)
        nu = eccentric_to_true(E, e)
    elif anomaly_type == "true":
        nu = anomaly
        E = true_to_eccentric(anomaly, e) # do the reverse
    else:
        raise ValueError("anomaly_type must be 'mean' or 'true'")
    

    # Step 2: Find orbital radius
    r_c = a * (1 - e * math.cos(E))

    # Step 3: Find the position and velocity vector in the orbital frame
    pos_o = r_c * np.array([math.cos(nu), math.sin(nu), 0])
    vel_o = (math.sqrt(mu / a) / (1 - e * math.cos(E))) * np.array([-math.sin(E), math.sqrt(1 - e**2) * math.cos(E), 0])

    # Step 4: Transform the position and velocity vector to the intertial frame.
    if frame == "inert": 
        NO_dcm = RTN_to_inert_313(RAAN, inc, aop)
        pos_i = NO_dcm @ pos_o
        vel_i = NO_dcm @ vel_o
    else:
        pos_i = pos_o
        vel_i = vel_o

    return pos_i, vel_i

def cartesian_to_keplerian(r_vec, v_vec, mu):
    """
    Convert Cartesian Coordinates in the inertial frame to the Keplerian Orbital Elements

    Parameters:
    x : float
        x position component
    y : float
        y position component
    z : float
        z position component
    xdot : float
        x velocity component
    ydot : float
        y velocity component
    zdot : float
        z velocity component
    mu : float
        Gravitational Parameter [km^3/s^2]

    Output:
    a : float
        semi-major axis [km]
    e : float
        eccentricity 
    i : float
        inclination 
    RAAN : float
        Right Ascension of the Ascending Node [rad]
    aop : float
        Argument of Periapsis [rad]
    M : float
        Mean Anomaly [rad]
    """
    # Step 1: Calculate Angular Momentum
    h_vec = np.cross(r_vec, v_vec)

    # Step 2: Calculate eccentricity vector
    r_hat = r_vec / np.linalg.norm(r_vec)
    e_vec = (np.cross(v_vec, h_vec) / mu) - r_hat

    # Step 3: Node vector (pointing toward ascending node)
    k_vec = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)

    # Step 4: Find orbit eccentricity
    e = np.linalg.norm(e_vec)

    # Step 5: Find true anomaly
    cos_nu = np.dot(e_vec, r_vec) / (e * np.linalg.norm(r_vec))
    nu = math.acos(np.clip(cos_nu, -1.0, 1.0))
    if np.dot(r_vec, v_vec) < 0:
        nu = 2 * np.pi - nu

    # Step 6: Find Eccentric Anomaly
    E = true_to_eccentric(nu, e)

    # Step 7: Find Mean Anomaly
    M = kepler_eqn(E, e)

    # Step 8: Find Inclination using inverse dcm mapping
    inc = np.acos(h_vec[2] / np.linalg.norm(h_vec))

    # Step 9: Find RAAN
    RAAN = np.acos(n_vec[0] / n)
    if n_vec[1] < 0:
        RAAN = 2 * np.pi - RAAN

    # Step 9: Find Argument of Periapsis 
    cos_aop = np.dot(n_vec, e_vec) / (n * np.linalg.norm(e_vec))
    aop = np.arccos(cos_aop)
    if e_vec[2] < 0:
        aop = 2 * np.pi - aop

    # Step 10: Find Semi-major axis
    a = 1 / ( (2 / np.linalg.norm(r_vec)) - ((np.linalg.norm(v_vec)**2) / mu) )  

    return a, e, inc, RAAN, aop, M

def equinoctal_to_keplerian(p, f, g, h, k, L):
    # semi-major axis
    a = p / (1 - f**2 - g**2)
    # orbital eccentricity
    e = np.sqrt(f**2 + g**2)
    # orbital inclination
    inc = np.atan2(2*np.sqrt(h**2 + k**2, 1 - h**2 - k**2))
    # Argument of periapsis
    aop = np.atan2(g*h - f*k, f*h + g * k)
    # Right ascension of the ascending node
    RAAN = np.atan2(k, h)
    # True anomaly
    nu = L - np.atan2(k, h)
    return a, e, inc, aop, RAAN, nu

def equinoctial_to_cartesian(p, f, g, h, k, L, mu, frame="inert"):
    """
    Convert equinotical orbital elements to keplerian orbit elments. 
    Parameters : 
    p : float
        semi-latus rectum [km]
    f : float

    g : float

    h : float

    k : float 

    L : float

    mu : float
        Gravitational Parameter
    
        
    """
    # a, e, inc, aop, RAAN, nu = equinoctal_to_keplerian(p, f, g, h, k, L)
    alpha2 = h**2 - k**2
    s2 = 1 + h**2 + k**2
    w = 1 + f * np.cos(L) + g * np.sin(L)
    r = p / w

    # position vector r
    r_vec = np.array([
        (r / s2) * (np.cos(L) + alpha2 * np.cos(L) + 2*h*k*np.sin(L)),
        (r / s2) * (np.sin(L) - alpha2 * np.sin(L) + 2*h*k*np.cos(L)),
        (2*r / s2) * (h*np.sin(L) - k*np.cos(L))
    ])

    # velocity vector v
    v_vec = np.array([
        -(1 / s2) * np.sqrt(mu / p) *
        (np.sin(L) + alpha2*np.sin(L) - 2*h*k*np.cos(L) + g - 2*f*h*k + alpha2*g),

        -(1 / s2) * np.sqrt(mu / p) *
        (-np.cos(L) + alpha2*np.cos(L) + 2*h*k*np.sin(L) - f + 2*g*h*k + alpha2*f),

        (2 / s2) * np.sqrt(mu / p) *
        (h*np.cos(L) + k*np.sin(L) + f*h + g*k)
    ])
    
    return r_vec, v_vec

def cartesian_to_equinoctial(r_vec, v_vec, mu):
    a, e, inc, RAAN, aop, M = cartesian_to_keplerian(r_vec, v_vec, mu)
    # Semi-parameter
    p = a * (1 - e**2)
    f = e * np.cos(aop + RAAN)
    g = e * np.sin(aop + RAAN)
    h = np.tan(inc/2) * np.cos(RAAN)
    k = np.tan(inc/2) * np.sin(RAAN)
    # Convert Mean Anomaly to True Anomaly
    E = solve_kepler(M, e)
    nu = eccentric_to_true(E, e) 
    # True Longitude
    L = RAAN + aop + nu
    L = np.mod(L, 2 * np.pi)
    return p, f, g, h, k, L


def milankovitch_to_cartesian(h_vec, e_vec, L, mu, frame="inert"):
    # Node Vector
    k_vec = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)

    # Find orbit eccentricity
    e = np.linalg.norm(e_vec)

    # Angular Momentum
    h = np.linalg.norm(h_vec)

    #  Inclination using inverse dcm mapping
    inc = np.acos(h_vec[2] / h)

    # RAAN
    if n > 1e-12:
        RAAN = np.arctan2(n_vec[1], n_vec[0])
    else:
        RAAN = 0.0
    RAAN = np.mod(RAAN, 2*np.pi)

    # Argument of periapsis
    if e > 1e-10 and n > 1e-10:
        aop = np.atan2(
            np.dot(np.cross(n_vec, e_vec), h_vec) / (n * h),
            np.dot(n_vec, e_vec) / n
        )
    else:
        aop = 0.0
    aop = np.mod(aop, 2*np.pi)


    # True anomaly
    nu = L - RAAN - aop
    nu = np.mod(nu, 2*np.pi)

    # Eccentric Anomaly
    E = true_to_eccentric(nu, e)

    # Mean Anomaly
    M = kepler_eqn(E, e)

    # Semi-latus rectum
    p = (np.linalg.norm(h_vec)**2) / mu

    # Semi-major axis
    a = p / (1 - e**2)

    r_vec, v_vec = keplerian_to_cartesian(a, e, inc, RAAN, aop, M, mu, frame=frame, anomaly_type="mean")

    return r_vec, v_vec


def cartesian_to_milankovitch(r_vec, v_vec, mu): 
    
    a, e, inc, RAAN, aop, M = cartesian_to_keplerian(r_vec, v_vec, mu)
    
    # Calculate Angular Momentum Vector
    h_vec = np.cross(r_vec, v_vec)

    # Calculate eccentricity vector
    r_hat = r_vec / np.linalg.norm(r_vec)
    e_vec = (np.cross(v_vec, h_vec) / mu) - r_hat

    # Convert Mean Anomaly to True Anomaly
    E = solve_kepler(M, e)
    nu = eccentric_to_true(E, e) 
    
    # true longitude
    L = RAAN + aop + nu
    L = np.mod(L, 2 * np.pi)
    
    return h_vec, e_vec, L



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
