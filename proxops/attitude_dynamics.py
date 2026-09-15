import numpy as np

def generate_pointing_attitude_reference(x_C_hist, x_T_hist, d=np.array([1.0, 0.0, 0.0])):
    """
    Generate a reference attitude trajectory where the camera axis d
    always points from the chaser toward the target.
    
    Args:
        x_C_hist: (13, N+1) chaser state history (position used from this)
        x_T_hist: (13, N+1) target state history
        d: camera boresight in body frame, default [1,0,0]
    
    Returns:
        q_ref:  (4, N+1) quaternions [q1, q2, q3, q4] where q4 is scalar
        w_ref:  (3, N+1) angular velocity in body frame
    """
    N = x_C_hist.shape[1] - 1
    q_ref = np.zeros((4, N+1))
    w_ref = np.zeros((3, N+1))

    for k in range(N+1):
        r_C = x_C_hist[0:3, k]
        r_T = x_T_hist[0:3, k]

        # Desired pointing direction in inertial frame
        rho = r_T - r_C
        rho_hat = rho / np.linalg.norm(rho)  # unit vector to target in inertial frame

        # We want C_IC such that C_IC @ rho_hat = d
        # i.e. rotate inertial rho_hat into body d
        # Use the rotation from rho_hat to d
        q_ref[:, k] = quat_from_two_vectors(rho_hat, d)

    # Compute angular velocity from finite differences of quaternion
    dt = 1  # matches your dt = 1
    for k in range(N):
        q_k   = q_ref[:, k]
        q_kp1 = q_ref[:, k+1]

        # Ensure quaternions are on the same hemisphere (avoid sign flips)
        if np.dot(q_k, q_kp1) < 0:
            q_kp1 = -q_kp1
            q_ref[:, k+1] = q_kp1

        # Quaternion derivative: q_dot ≈ (q_{k+1} - q_k) / dt
        q_dot = (q_kp1 - q_k) / dt

        # Invert kinematic equation: q_dot = 0.5 * Omega(w) @ q
        # => w = 2 * Xi(q)^T @ q_dot  where Xi is the 4x3 left matrix
        q1, q2, q3, q4 = q_k
        Xi = np.array([
            [ q4, -q3,  q2],
            [ q3,  q4, -q1],
            [-q2,  q1,  q4],
            [-q1, -q2, -q3]
        ])
        w_ref[:, k] = 2.0 * Xi.T @ q_dot

    # Last angular velocity = second to last (simple extrapolation)
    w_ref[:, -1] = w_ref[:, -2]

    return q_ref, w_ref


def quat_from_two_vectors(v_from, v_to):
    """
    Compute quaternion [q1,q2,q3,q4] (scalar-last) that rotates
    unit vector v_from to unit vector v_to.
    """
    v_from = v_from / np.linalg.norm(v_from)
    v_to   = v_to   / np.linalg.norm(v_to)

    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)

    # Parallel case: no rotation needed
    if dot > 1.0 - 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])

    # Anti-parallel case: 180 deg rotation about any perpendicular axis
    if dot < -1.0 + 1e-10:
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(v_from, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(v_from, perp)
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])  # 180 deg rotation

    # General case
    axis = np.cross(v_from, v_to)
    axis_norm = np.linalg.norm(axis)
    axis /= axis_norm

    angle = np.arccos(dot)
    s = np.sin(angle / 2)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle / 2)])

def ep_kde(omega, q):
    wx, wy, wz = omega
    Omega = np.array([
        [0, wz, -wy, wx],
        [-wz, 0, wx, wy],
        [wy, -wx, 0, wz],
        [-wx, -wy, -wz, 0]
    ])
    q_dot = 0.5 * Omega @ q

    return q_dot

def L_gg(r_sc, mu, I_s):
    Ixx = I_s[0, 0]
    Iyy = I_s[1, 1]
    Izz = I_s[2, 2]
    rx = r_sc[0]
    ry = r_sc[1]
    rz = r_sc[2]
    r_norm = np.linalg.norm(r_sc)
    L_gg = ((3 * mu) / r_norm**5) * np.array([(Iyy - Izz)*ry*rz, (Izz - Ixx)*rz*rx, (Ixx - Iyy)*rx*ry])
    return L_gg

def ep2dcm(ep):
    q1 = ep[0]
    q2 = ep[1]
    q3 = ep[2]
    q4 = ep[3]
    # Compute the Direction Cosine Matrix (DCM) from the Euler Parameters
    DCM = np.array([[1 - 2*q2**2 - 2*q3**2, 2*(q1*q2 + q3*q4), 2*(q1*q3 - q2*q4)],
        [2*(q1*q2 - q3*q4), 1 - 2*q1**2 - 2*q3**2, 2*(q2*q3 + q1*q4)],
        [2*(q1*q3 + q2*q4), 2*(q2*q3 - q1*q4), 1 - 2*q1**2 - 2*q2**2]])
    return DCM

def dde(I_mat, w, tau):
    I_inv = np.diag([(1/I_mat[0][0]), (1/I_mat[1][1]), (1/I_mat[2][2])])
    I_omega = I_mat @ w
    w_dot = I_inv @ (tau - np.cross(w, I_omega)) # needs to be a vector

    return w_dot

def crossmat(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x_til = np.array([[0, -x3, x2], [x3, 0, -x1], [-x2, x1, 0]])
    return x_til


def compute_fov_full(x_opt, x_T_hist, b, d):
    beta = np.zeros(x_opt.shape[1])
    for i in range(x_opt.shape[1]):
        beta[i] = compute_fov(x_opt[:, i], x_T_hist[:, i], b, d)

    return beta

def compute_fov_constraint_nl(x_opt, x_T_hist, b, d, beta_max):
    I_rho = x_opt[0:3] - x_T_hist[0:3]
    C_IC = ep2dcm(x_opt[6:10])
    C_rho = C_IC @ I_rho
    g = np.dot((C_rho + b), d) + np.linalg.norm(C_rho + b) * np.cos(beta_max)
    return g

def compute_fov(x_opt, x_T_hist, b, d):
    I_rho = x_opt[0:3] - x_T_hist[0:3]
    C_IC = ep2dcm(x_opt[6:10])
    C_rho = C_IC @ I_rho
    inner = ( -np.dot((C_rho + b), d) / np.linalg.norm(C_rho + b) )
    beta = np.acos(inner)
    return beta