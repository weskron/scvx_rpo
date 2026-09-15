import numpy as np
from proxops.orbital_frame_conversions import *


def generate_nonlin_traj(x_0, I_mat, u_hist, tf, N):

    # Initialize time-step
    dt = int(tf/N)

    # Initialize variable to store state history
    x_hist = np.zeros((13, N+1))

    # Set initial state
    x_k = x_0
    x_hist[:, 0] = x_0

    # Solve nonlinear dynamics at each time-step
    for k in range(N):
        tk = k * dt
        tkp1 = (k+1) * dt

        traj = solve_ivp(
            nl_dyn,
            [tk, tkp1],
            x_k,
            args=(I_mat, u_hist[:, k]),
            atol=tol, rtol=tol,
        )

        # Store result
        x_hist[:, k+1] = traj.y[:, -1]

        # Set initial state for next step
        x_k = x_hist[:, k+1]

    return x_hist

def nl_dyn(t, x, I_mat, u_hist):
    # unpack state
    r = x[0:3]
    v = x[3:6]
    q = x[6:10]
    w = x[10:13]

    u_accel = u_hist[0:3]
    tau = u_hist[3:6]

    # Helpers
    r_norm = np.linalg.norm(r)

    # Compute Derivative
    r_dot = v

    v_dot = -(mu/r_norm**3) * r + u_accel

    q_dot = ep_kde(w, q)

    w_dot = dde(I_mat, w, tau)

    x_dot = np.concatenate([r_dot.flatten(), v_dot.flatten(), q_dot.flatten(), w_dot.flatten()])

    return x_dot


