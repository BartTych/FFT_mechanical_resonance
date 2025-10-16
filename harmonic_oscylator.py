
import numpy as np
from matplotlib import pyplot as plt


# parameters
k = 100_000.0     # stiffness [N/m]
c = 15.0          # damping [N·s/m]
m = 1.0           # mass [kg]

# state
u = np.array([0.0, 0.0])   # [base, mass] displacements
v = np.zeros_like(u)
a = np.zeros_like(u)

# integration parameters
dt = 1e-6
steps = int(1.3 * 1e5)     # 130000 steps
T = 0.0

# excitation parameters
exc_amp = 0.002            # 2 mm amplitude
exc_freq = 18.82            # 40 Hz
omega = 2*np.pi*exc_freq

# storage
x_log = np.zeros(steps)
v_log = np.zeros(steps)
t_log = np.zeros(steps)

x_log_rotated = np.zeros(steps)
v_log_rotated = np.zeros(steps)

for n in range(steps):
    # excitation 
    u[0] = exc_amp * np.sin(2*np.pi*exc_freq*T)
    v[0] = 2*np.pi*exc_freq*exc_amp * np.cos(2*np.pi*exc_freq*T)

    # relative motion between mass and base
    rel_u = u[1] - u[0]
    rel_v = v[1] - v[0]

    # spring + damping forces
    F_spring = k * rel_u
    F_damp   = c * rel_v
    F_total  = -(F_spring + F_damp)  # force on the mass

    # acceleration of mass
    a[1] = F_total / m

    # explicit Euler integration
    v[1] = v[1] + dt * a[1]
    u[1] = u[1] + dt * v[1]

    # log
    x_log[n] = u[1]
    v_log[n] = v[1]
    t_log[n] = T

    phi = omega * T
    
    x = u[1]
    v_scaled = v[1] / omega

    z_rot = (x + 1j*v_scaled) * np.exp(1j*phi)
    x_log_rotated[n] = z_rot.real
    v_log_rotated[n] = z_rot.imag

    # advance time
    T += dt

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(9, 9))
scale = 1e-4
ax.plot(v_log_rotated / scale, x_log_rotated / scale)


ax.plot(x_log_rotated, v_log_rotated, lw=1.0)
ax.set_ylabel('x [×10⁻⁴ m]')
ax.set_xlabel('v [×10⁻⁴/2*pi m/s]')# after rotation units are becoming hard to interpret 
ax.set_title('Phase Space: x–v')

ax.set_aspect('equal', adjustable='box')
plt.show()