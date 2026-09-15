import sympy as sp
import numpy as np
# Symbols
rx, ry, rz = sp.symbols('rx ry rz')
vx, vy, vz = sp.symbols('vx vy vz')
q0, q1, q2, q3 = sp.symbols('q0 q1 q2 q3')
wx, wy, wz = sp.symbols('wx wy wz')
rxT, ryT, rzT = sp.symbols('rxT ryT rzT')
b1, b2, b3 = sp.symbols('b1 b2 b3')
d1, d2, d3 = sp.symbols('d1 d2 d3')
beta_max = sp.Symbol('beta_max')

x = sp.Matrix([rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz])

# DCM (q0 scalar first)
C = sp.Matrix([
    [1-2*q1**2-2*q2**2,   2*(q0*q1+q2*q3),   2*(q0*q2-q1*q3)],
    [2*(q0*q1-q2*q3),     1-2*q0**2-2*q2**2, 2*(q1*q2+q0*q3)],
    [2*(q0*q2+q1*q3),     2*(q1*q2-q0*q3),   1-2*q0**2-2*q1**2]
])

I_rho = sp.Matrix([rx - rxT, ry - ryT, rz - rzT])
b_vec = sp.Matrix([b1, b2, b3])
d_vec = sp.Matrix([d1, d2, d3])

C_rho = C @ I_rho
vec   = C_rho + b_vec

g = sp.Matrix([vec.dot(d_vec) - sp.sqrt(vec.dot(vec)) * sp.cos(beta_max)])

g = g.jacobian(x)

inputs = (
    rx,  ry,  rz,
    vx,  vy,  vz,
    q0,  q1,  q2,  q3,
    wx,  wy,  wz,
    rxT, ryT, rzT,
    b1,  b2,  b3,
    d1,  d2,  d3,
    beta_max,
)

g_func   = sp.lambdify(inputs, g,   'numpy')

def _pack(x_val, x_T_val, b_val, d_val, beta_max_val):
    return (
        x_val[0],  x_val[1],  x_val[2],
        x_val[3],  x_val[4],  x_val[5],
        x_val[6],  x_val[7],  x_val[8],  x_val[9],
        x_val[10], x_val[11], x_val[12],
        x_T_val[0], x_T_val[1], x_T_val[2],
        b_val[0],  b_val[1],  b_val[2],
        d_val[0],  d_val[1],  d_val[2],
        beta_max_val,
    )

def compute_fov_constraint(x_val, x_T_val, b_val, d_val, beta_max_val):
    return (g_func(*_pack(x_val, x_T_val, b_val, d_val, beta_max_val)))