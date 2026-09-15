import numpy as np

def lin_discrete(x_ref, I_mat, u_hist, tf, N, n_x, n_u, mu):

    # Get time-step 
    dt = int(tf / N)

    # Initialize discrete matrices
    A_list = []
    B_list = []
    c_list = []

    # Loop through all k
    for k in range(N):
        
        # Solver time information
        t_k = k * dt
        t_kpls1 = (k + 1) * dt

        # Setup numerical integration
        tol = 1e-12

        # Pack conditions at current k into augmented state vector
        Y_k = np.concatenate([np.eye(n_x).flatten(), np.zeros((n_x * n_u)).flatten()])

        # Perform numerical integration
        res = solve_ivp(
            discrete_eoms, 
            [t_k, t_kpls1], 
            Y_k, 
            args=(x_ref[:, k], I_mat, n_x, n_u, mu), 
            rtol=tol, atol=tol)

        # Unpack augmented state vector at k+1
        A_k = res.y[0:n_x**2, -1].reshape(n_x, n_x)
        B_k = res.y[n_x**2:(n_x**2 + n_x*n_u), -1].reshape(n_x, n_u)

        # # Calculate c_k using reference trajectory
        c_k = x_ref[:, k+1] - A_k @ x_ref[:, k] - B_k @ u_hist[:, k]

        # # Store A_k, B_k, and c_k
        A_list.append(A_k)
        B_list.append(B_k)
        c_list.append(c_k)

    return A_list, B_list, c_list

def discrete_eoms(t, Y, x_ref, I_mat, n_x, n_u, mu):
    
    # Unpack augmented state vector
    phi = Y[0: (n_x**2)].reshape(n_x, n_x)
    phi_B = Y[(n_x**2):(n_x**2 + n_x * n_u)].reshape((n_x, n_u))

    # Calculate A matrix 
    A = compute_jacobian(x_ref, mu, np.array([I_mat[0][0], I_mat[1][1], I_mat[2][2]]))
    
    # Establish B matrix
    I_inv = np.diag([1/I_mat[0,0], 1/I_mat[1,1], 1/I_mat[2,2]])

    B = np.block([[np.zeros((3, 3)), np.zeros((3, 3))],
                [np.eye(3),        np.zeros((3, 3))],
                [np.zeros((4, 3)), np.zeros((4, 3))],
                [np.zeros((3, 3)), I_inv  ]])
    
    # Build derivative of augmented state vector
    phi_dot = A @ phi
    phi_B_dot = A @ phi_B + B

    Y_dot = np.concatenate([phi_dot.flatten(), phi_B_dot.flatten()])
    
    return Y_dot
