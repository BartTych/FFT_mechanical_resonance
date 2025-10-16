import numpy as np
import os
from matplotlib import pyplot as plt

# output folder
output_dir = "jeffcott_plots"
os.makedirs(output_dir, exist_ok=True)

#f_spin frequency of rotation
# loop makes plot for each frequency
for f_spin in np.linspace(3,57,100):

    # parameters
    k   = 100_000         # shaft stiffness [N/m]
    c   = 15                # viscous damping

    m_r = 1.0                       # rotating (disk) mass [kg]
    e   = 0.002                      # eccentricity [m]
    #f_spin = 26 # used for run with one frequency                
    Omega  = 2*np.pi*f_spin         # spin speed [rad/s]

    # state
    u = np.zeros(2)   # [ux, uy]
    v = np.zeros(2)   # [vx, vy]
    a = np.zeros(2)   # [ax, ay]

    # integration setup
    dt    = 2e-5
    tf    = 2.0
    steps = int(tf/dt)
    T     = 0.0

    # logs
    t_log  = np.zeros(steps)
    ux_log = np.zeros(steps); uy_log = np.zeros(steps)
    vx_log = np.zeros(steps); vy_log = np.zeros(steps)

    # rotated-frame logs (unit-consistent: v/Ω)
    ux_rot = np.zeros(steps); uy_rot = np.zeros(steps)

    for n in range(steps):
        # log current state
        t_log[n]  = T
        ux_log[n] = u[0]; uy_log[n] = u[1]
        vx_log[n] = v[0]; vy_log[n] = v[1]

        # unbalance force components: F = m_r * e * Ω^2 [cos Ωt, sin Ωt]
        cosO = np.cos(Omega*T);  sinO = np.sin(Omega*T)
        F = (m_r * e * Omega**2) * np.array([cosO, sinO])

        # acceleration: m*u'' + c*u' + k*u = F
        a = (F - c*v - k*u) / m_r

        # explicit Euler update
        v += dt * a
        u += dt * v

        # synchronous rotating frame (response collapse to a point in steady state)
        z  = (u[0] + 1j*u[1])                 # position
        rot = np.exp(-1j*Omega*T)              
        zr = z * rot
        ux_rot[n], uy_rot[n] = zr.real, zr.imag
        
        # advance time
        T += dt

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 9))
    scale = 1e-4
    ax.plot(ux_rot / scale, uy_rot / scale)
    ax.set_xlabel('x [×10⁻⁴ m]')
    ax.set_ylabel('y [×10⁻⁴ m]')
    ax.plot(ux_rot, uy_rot, lw=1.0)

    #ax.set_xlabel('x (m)')
    #ax.set_ylabel('y (m)')
    ax.set_title(f'Response in rotating frame {f_spin:.2f} Hz')
    #ax.grid(True, alpha=0.3)

    ax.set_aspect('equal', adjustable='box')
    filepath = os.path.join(output_dir, f"Response_{f_spin:.2f} Hz.jpg")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    #plt.show()