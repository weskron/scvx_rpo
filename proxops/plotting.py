import numpy as np
import matplotlib.pyplot as plt

def plot_fov(tf, beta, max_fov, N):
    plt.figure(6)
    t = np.linspace(0, tf, N+1)
    plt.plot(t, np.rad2deg(beta))
    plt.hlines(np.rad2deg(max_fov), 0, tf)
    plt.grid()

def plot_penalty(t, ptr_opt):
    plt.figure(4)
    plt.plot(t, ptr_opt)
    plt.title("Penalty Constraint Over Time")

def plot_virtual_control(t, v_cp):
    plt.figure(5)
    plt.plot(t, np.linalg.norm(v_cp.value, axis=0))
    plt.title("Translation Virtual Controller Over Time")


def plot_trajectory(x_hist, plot_label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_hist[0, :], x_hist[1, :], x_hist[2,:], 'b-', label='Spacecraft')
    ax.plot(x_hist[0, 0],  x_hist[1, 0], x_hist[2, 0], 'g^', markersize=10, label='Start')
    ax.plot(x_hist[0, -1], x_hist[1, -1], x_hist[2,-1], 'r^', markersize=10, label='End')

    max = np.max(x_hist)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-max, max])
    ax.set_ylim([-max, max])
    ax.set_zlim([-max, max])
    ax.set_title(plot_label)
    #ax.set_aspect('equal')
    ax.legend()

def plot_attitude(tf, epsilon, tar_epsilon, omega, plot_label): 
    t = np.linspace(0, tf, epsilon.shape[1])

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, epsilon[0, :], label="chaser $q_1$")
    ax[0].plot(t, epsilon[1, :], label="chaser $q_2$")
    ax[0].plot(t, epsilon[2, :], label="chaser $q_3$")
    ax[0].plot(t, epsilon[3, :], label="chaser $q_4$")
    ax[0].plot(t, tar_epsilon[0, :], label="Target $q_1$")
    ax[0].plot(t, tar_epsilon[1, :], label="Target $q_2$")
    ax[0].plot(t, tar_epsilon[2, :], label="Target $q_3$")
    ax[0].plot(t, tar_epsilon[3, :], label="Target $q_4$")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_title("Epsilon")


    ax[1].plot(t, omega[0, :])
    ax[1].plot(t, omega[1, :])
    ax[1].plot(t, omega[2, :])
    ax[1].grid(True)
    ax[1].set_title("Angular Velocity")

    fig.suptitle(plot_label)

    return True

def plot_translation_controls(u_hist, u_max, dt):
    N = u_hist.shape[1]
    t = np.arange(N) * dt / 60.0
    norms = np.linalg.norm(u_hist, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, u_hist[0, :], color='tab:red',   linewidth=1.4, label='$u_x$ (radial)')
    ax.plot(t, u_hist[1, :], color='tab:green', linewidth=1.4, label='$u_y$ (along-track)')
    ax.plot(t, u_hist[2, :], color='tab:blue',  linewidth=1.4, label='$u_z$ (cross-track)')
    ax.plot(t, norms,        color='black',     linewidth=1.8, label='$\|u\|_2$', linestyle='-.')
    ax.axhline( u_max, color='black', linewidth=1.0, linestyle='--', label='$u_{max}$')
    ax.axhline(-u_max, color='black', linewidth=1.0, linestyle='--')

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Optimal Control History')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

def plot_attitude_control(tf, torque, u_r_max):
    t = np.linspace(0, tf, torque.shape[1])
    norms = np.linalg.norm(torque, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, torque[0, :], color='tab:red',   linewidth=1.4, label='$\tau_x$')
    ax.plot(t, torque[1, :], color='tab:green', linewidth=1.4, label='$\tau_y$')
    ax.plot(t, torque[2, :], color='tab:blue',  linewidth=1.4, label='$\tau_z$')
    ax.plot(t, norms,        color='black',     linewidth=1.8, label='$\|u\|_2$', linestyle='-.')
    ax.axhline( u_r_max, color='black', linewidth=1.0, linestyle='--', label='$u_{max}$')
    ax.axhline(-u_r_max, color='black', linewidth=1.0, linestyle='--')

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Attitude Optimal Control History')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
