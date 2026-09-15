import sys
import os
# Add the project root to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import sympy as sp

# General Symbols
mu = sp.symbols('mu')
Ix, Iy, Iz = sp.symbols('Ix Iy Iz')

# Position
rx, ry, rz = sp.symbols('rx ry rz')

# Velocity
vx, vy, vz = sp.symbols('vx vy vz')

# Quaternion (Euler parameters)
q0, q1, q2, q3 = sp.symbols('q0 q1 q2 q3')

# Angular velocity
wx, wy, wz = sp.symbols('wx wy wz')

# State vector
x = sp.Matrix([rx, ry, rz,
               vx, vy, vz,
               q0, q1, q2, q3,
               wx, wy, wz])

# Example placeholders (replace with your real equations)

# Position derivative = velocity
r_dot = sp.Matrix([vx, vy, vz])

# Velocity derivative (example: acceleration)
r = sp.Matrix([rx, ry, rz])
r_norm = sp.sqrt(r.dot(r))

a = -mu / r_norm**3 * r
v_dot = a

# Quaternion kinematics
Omega = sp.Matrix([
    [0, wz, -wy, wx],
    [-wz, 0, wx, wy],
    [wy, -wx, 0, wz],
    [-wx, -wy, -wz, 0]
])

q = sp.Matrix([q0, q1, q2, q3])
q_dot = 0.5 * Omega * q

# Angular acceleration (placeholder)
I = sp.diag(Ix, Iy, Iz)
omega = sp.Matrix([wx, wy, wz])

w_dot = I.inv() * (-omega.cross(I * omega))


# Full dynamics
f = sp.Matrix.vstack(r_dot, v_dot, q_dot, w_dot)

J = f.jacobian(x)

inputs = (
    rx, ry, rz,
    vx, vy, vz,
    q0, q1, q2, q3,
    wx, wy, wz,
    mu, Ix, Iy, Iz
)

J_func = sp.lambdify(inputs, J, 'numpy')

def compute_jacobian(x_val, mu_val, I_val):
    return J_func(
        x_val[0], x_val[1], x_val[2],
        x_val[3], x_val[4], x_val[5],
        x_val[6], x_val[7], x_val[8], x_val[9],
        x_val[10], x_val[11], x_val[12],
        mu_val,
        I_val[0], I_val[1], I_val[2]
    )