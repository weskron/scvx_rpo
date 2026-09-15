import sys
import os
# Add the project root to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from proxops.jacobian import compute_jacobian
from proxops.fov_jacobian import compute_fov_constraint
from scipy.integrate import solve_ivp
from proxops.orbital_frame_conversions import keplerian_to_cartesian

mu = 3.98600e8
u_max = 0.5 # m/s2
tau_max = 3*22 # N-m
tol = 1e-12 # Integration Tolerance

def main():
    # Time
    tf = 5 * 60
    dt = 1
    N = int(tf/dt)

    n_x = 13
    n_u = 6
    
    # -- Target Initalizations ---------------
    m_tar = 500 # kg
    l_s, w_s, h_s = 3, 3, 3 # cube
    I_tar = (1/12) * m_tar * np.diag([
        w_s**2 + h_s**2,
        l_s**2 + h_s**2,
        l_s**2 + w_s**2
    ])
    C_IL = []

    tar_a = 7000e3 # km
    tar_inc = 0 
    tar_RAAN = 0
    tar_aop = 0
    tar_nu = 0
    r_T_0, v_T_0 = keplerian_to_cartesian(tar_a, 0, tar_inc, tar_RAAN, tar_aop, tar_nu, mu=mu, frame="inert", anomaly_type="mean")
    theta_T = np.deg2rad(4)
    lambda_hat = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
    q_T_0 = np.array([1/np.sqrt(2) * np.sin(theta_T), 1/np.sqrt(2) * np.sin(theta_T), 0, np.cos(theta_T)])
    w_T_0 = np.array([0.01, 0.00, 0.0])  
    x_T_init = np.concatenate([r_T_0, v_T_0, q_T_0, w_T_0])


    # -- Chaser Initializations --------------
    m_sat = 800 # kg
    l_s, w_s, h_s = 3, 3, 3 # cube
    I_cha = (1/12) * m_sat * np.diag([
        w_s**2 + h_s**2,
        l_s**2 + h_s**2,
        l_s**2 + w_s**2
    ])
    r_C_0, v_C_0 = keplerian_to_cartesian(tar_a, 0, tar_inc, tar_RAAN, tar_aop, 1.42857e-4, mu=mu, frame="inert", anomaly_type="mean") # put 1000 m in front of target
    rho_0 = r_C_0 - r_T_0
    lambda_hat = np.cross(rho_0, np.array([1, 0, 0]))
    theta = np.acos(np.dot(rho_0, np.array([1, 0, 0])) / np.linalg.norm(rho_0))
    # q_C_0 = np.array([lambda_hat[0]* np.sin(theta / 2), lambda_hat[1]* np.sin(theta / 2), lambda_hat[2] * np.sin(theta / 2), np.cos(theta / 2)])
    q_C_0 = np.array([0.0, 0.0, 0.0, 1.0])
    w_C_0 = np.array([0.0, 0.0, 0.0])    
    x_C_init = np.concatenate([r_C_0, v_C_0, q_C_0, w_C_0])


    # -- Compute Target History --------------
    u_hist = np.zeros((n_u, N))
    x_T_hist = generate_nonlin_traj(x_T_init, I_tar, u_hist, tf, N)
    print("Target Traj Generated")

    # -- Compute Initial Chaser Nonlinear Traj ----
    x_C_hist = generate_nonlin_traj(x_C_init, I_cha, u_hist, tf, N) # nl reference
    print("Chaser Traj Generated")

    # -- Check Target History and Initial Chaser Reference

    # -- Generate Reference Chaser Trajectory ------
    # position is a straight line
    r_C_ref = r_C_0 + np.linspace(0, 1, N+1)[:, None] * (r_T_0 - r_C_0)
    r_C_ref = r_C_ref.T  # shape (3, N+1)
    # velocity is a constant value 0
    v_C_ref = np.zeros((3, N+1))
    # quat is ...
    q_C_hist = x_C_hist[6:10, :]
    # omega is ...
    w_C_hist = x_C_hist[10:13, :]

    # Build a temporary x_C_hist using the straight-line position reference
    # so the pointing is computed from the actual reference positions
    # d = np.array([1, 0, 0])
    # x_C_pos_ref = np.vstack([r_C_ref, v_C_ref, np.zeros((9, N+1))])
    # q_C_hist, w_C_hist = generate_pointing_attitude_reference(x_C_pos_ref, x_T_hist, d=d)
    
    # -- Compute Initial Lin Disc ------------
    # Create a function that takes in the reference and computes Ak Bk ck
    A_list, B_list, c_list = lin_discrete(x_C_hist, I_cha, u_hist, tf, N, n_x, n_u, mu) # TODO: Update later
    print("A B and c generated")
    x_lin_hist = np.zeros((n_x, N+1))
    x_lin_hist[:, 0] = x_C_hist[:, 0]

    for i in range(N):
        x_lin_hist[:, i+1] = A_list[i] @ x_lin_hist[:, i] + B_list[i] @ u_hist[:, i] + c_list[i]

    # plot_trajectory(x_lin_hist[:3, :], "Linearized Chaser")
    # plot_attitude(tf, x_lin_hist[6:10, :], x_lin_hist[10:13, :], "Linearized Chaser")

    # -- FOV Initialization ------------------
    b = np.array([1.5, 0.0, 0.0])
    d = np.array([1.0, 0.0, 0.0])
    beta_fov = np.deg2rad(60)
    fov_pause = 0
        

    # -- SCP Initialization ------------------
    iter = 0
    max_iter = 25

    lam1 = 1e0
    lam2 = 1e2
    lam3 = 1e3

    pos_scale = 1e4
    vel_scale = 10
    ep_scale = 1
    w_scale = 0.1
    alpha_x = np.diag([
                1/pos_scale, 1/pos_scale, 1/pos_scale,
                1/vel_scale, 1/vel_scale, 1/vel_scale,
                1/ep_scale, 1/ep_scale, 1/ep_scale, 1/ep_scale,
                1/w_scale, 1/w_scale, 1/w_scale])
    
    alpha_u = np.zeros((6, 6)) # np.diag([
    #     1/u_max, 1/u_max, 1/u_max,
    #     1/tau_max, 1/tau_max, 1/tau_max
    # ])

    x_C_ref = np.vstack([r_C_ref, v_C_ref, q_C_hist, w_C_hist])
    u_ref = u_hist
    ptr_opt = 100

    delta_k = 1
    rho0 = 0.1
    rho1 = 0.9

    # SCP
    while iter < max_iter:
        print(f"===== Running SCP Iteration {iter+1} ===== \n")

        # Initialize cp vars
        x_cp = cp.Variable((n_x, N+1))
        u_cp = cp.Variable((n_u, N))
        
        # Initialize slack vars for FOV
        chi_fov_cp = cp.Variable((N-fov_pause))

        # Initialize virtual control for nl dynamics
        v_cp = cp.Variable((13, N))
        
        # Initialize cost and constraint
        constraints = []
        J = 0

        # for loop over N
        for k in range(N):
            # Add dynamics 
            constraints += [x_cp[:, k+1] == A_list[k] @ x_cp[:, k] + B_list[k] @ u_cp[:, k] + c_list[k] + v_cp[:, k]]
            # Add control constraints
            constraints += [cp.norm(u_cp[0:3, k], 2) - u_max <= 0]
            constraints += [cp.norm(u_cp[3:6, k], 2) - tau_max <= 0]

            # Add Penalty Trust Region
            constraints += [cp.norm(alpha_x @ (x_cp[:, k] - x_C_ref[:, k]), 2) + cp.norm(alpha_u @ (u_cp[:, k] - u_ref[:, k]), 2) <= delta_k]

            # fov constraint
            if k >= fov_pause:
                g_ref = compute_fov_constraint(x_C_ref[:, k], x_T_hist[:, k], b, d, beta_fov)
                g = compute_fov_constraint_nl(x_C_ref[:, k], x_T_hist[:, k], b, d, beta_fov)
                constraints += [g + g_ref @ (x_cp[:, k] - x_C_ref[:, k]) <= chi_fov_cp[k-fov_pause]]
                constraints += [chi_fov_cp[k-fov_pause] >= 0]


        # Boundary Constraints
        constraints += [x_cp[0:6, 0] == x_C_ref[0:6, 0]] # Fix intial position
        constraints += [x_cp[6:13, 0] == x_C_ref[6:13, 0]] # Fix intial position
        constraints += [x_cp[:6, -1] == x_T_hist[:6, -1]] # Match target State
    
        # Create Cost function
        J = cp.sum([cp.norm(u_cp[:, k], 2) for k in range(N)]) * dt + lam2 * cp.sum([cp.norm(v_cp[:, k], axis=0) for k in range(N)]) + lam3 * cp.sum(cp.norm1(chi_fov_cp))

        # Solve Convex Problem
        print(f"Solving Convex Subproblem \n")
        objective = cp.Minimize(J)
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK)
        L_opt = J.value
        x_opt = x_cp.value
        u_opt = u_cp.value
        v_opt = v_cp.value
        
        # Output solver status
        print("Problem Status: ", prob.status)
        print("Optimal Cost:   ", prob.value)
        print("Virtual Control: ", np.linalg.norm(v_opt))

        # Skip evaluation of rho for first iteration so we can input arbitrary trajectory
        if iter == 0:
            # Reference trajectory and go to next iteration
            L_ref = L_opt
            x_C_ref = x_opt
            u_ref = u_opt
            
            # Genereate new Ak Bk ck based on new reference
            A_list, B_list, c_list = lin_discrete(x_C_ref, I_cha, u_ref, tf, N, n_x, n_u, mu)
        else: 
        # Set L_true
            L_true = L_opt

            # Compute J_true(x_ref)
            J_true_ref = J_true(x_C_ref[:13, 0], I_cha, x_C_ref, u_ref, tf, N, fov_pause, x_T_hist, b, d, beta_fov, lam2, lam3)
            
            # Compute J_true(x_opt)
            J_true_opt = J_true(x_C_ref[:13, 0], I_cha, x_opt, u_opt, tf, N, fov_pause, x_T_hist, b, d, beta_fov, lam2, lam3)
            
            # Compute rho
            delta_J = J_true_ref - J_true_opt
            delta_L = L_ref - L_true

            # SCVx acceptance logic - MAY NEED REWRITE
            if abs(delta_L) < 1e-10:
                rho = 1.0
            else:
                rho = delta_J / delta_L

            if rho < rho0:
                delta_k *= 0.5
                print(f"rho: {rho:.4f}, delta_J: {delta_J:.4f}, delta_L: {delta_L:.4f}, delta_k: {delta_k:.4f}")
                continue
            else:
                # Set reference information
                x_C_ref = x_opt
                u_ref = u_opt
                L_ref = L_opt

                # Genereate new Ak Bk ck based on new reference
                A_list, B_list, c_list = lin_discrete(x_C_ref, I_cha, u_ref, tf, N, n_x, n_u, mu)

                print(f"rho: {rho:.4f}, delta_J: {delta_J:.4f}, delta_L: {delta_L:.4f}, delta_k: {delta_k:.4f}")

                if rho > rho1:
                    delta_k *= 1.5

        
        iter += 1

    x_rel = x_C_ref[:3, :] - x_T_hist[:3, :]
    t = np.linspace(0, tf, N)

    g_true = []
    g_lin = []

    for k in range(fov_pause, N):
        # Nonlinear constraint at solution
        g_nl = compute_fov_constraint_nl(x_opt[:, k], x_T_hist[:, k], b, d, beta_fov)

        # Linearized constraint evaluated at solution
        g0 = compute_fov_constraint_nl(x_C_ref[:, k], x_T_hist[:, k], b, d, beta_fov)
        grad = compute_fov_constraint(x_C_ref[:, k], x_T_hist[:, k], b, d, beta_fov)
        g_l = g0 + grad @ (x_opt[:, k] - x_C_ref[:, k])

        g_true.append(g_nl)
        g_lin.append(g_l)

def J_true(x_ref_0, I_cha, x_opt, u, tf, N, fov_pause, x_T_hist, b, d, beta_fov, lam2, lam3):
    control = sum(np.linalg.norm(u[:, k]) for k in range(N))

    dyn = 0 
    x_true = generate_nonlin_traj(x_ref_0, I_cha, u, tf, N) # nl reference
    
    for k in range(N-1):
        dyn += np.linalg.norm(x_true[:, k+1] - x_opt[:, k+1])

    fov = 0
    for k in range(fov_pause, N):
        g = compute_fov_constraint_nl(x_true[:, k], x_T_hist[:, k], b, d, beta_fov)
        fov += max(0, g)

    return control + lam2 * dyn + lam3 * fov

if __name__ == "__main__":
    main()