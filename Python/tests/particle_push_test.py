import numpy as np
import matplotlib.pyplot as plt

def test_digital_twin_boris():
    # 1. Constants (Matching your DigitalTwinSimulator exactly)
    q = 1.602e-19
    m_XE = 131.293 * 1.6605e-27
    q_m = q / m_XE
    
    # 0.1 Tesla is very strong for Xenon, we'll use a strong field and large dt 
    # to see multiple cyclotron orbits.
    Bz = 0.5     
    dt = 1e-9    
    steps = 20000
    
    # Initial Conditions
    v0 = 1e4 # 10,000 m/s
    px, py = 0.0, 0.0
    vx, vy, vz = v0, 0.0, 0.0
    
    # 2. Analytical Calculations
    # Radius in meters: r = mv / qB
    r_analytical_m = (m_XE * v0) / (q * Bz)
    # Convert to mm to match your engine's coordinate system!
    r_analytical_mm = r_analytical_m * 1000.0 
    
    omega = (q * Bz) / m_XE # Cyclotron frequency (rad/s)
    
    traj_num_x = []
    traj_num_y = []
    
    # 3. Engine Simulation Loop
    for _ in range(steps):
        # We assume E-fields are 0 for a pure magnetic rotation test
        Ex_p = 0.0
        Ey_p = 0.0
        Bx_p = 0.0
        By_p = 0.0
        Bz_p = Bz
        
        # =====================================================================
        # EXACT REPLICA OF: push_particles_boris_taichi
        # =====================================================================
        
        # STEP 1: First half E-field acceleration (v_minus)
        v_minus_x = vx + (q_m * Ex_p * dt) / 2.0
        v_minus_y = vy + (q_m * Ey_p * dt) / 2.0
        v_minus_z = vz # Ez is 0

        # STEP 2: Magnetic Field Rotation (v_plus)
        t_x = (q_m * Bx_p * dt) / 2.0
        t_y = (q_m * By_p * dt) / 2.0
        t_z = (q_m * Bz_p * dt) / 2.0
        t_mag_sq = t_x**2 + t_y**2 + t_z**2
        
        s_x = 2.0 * t_x / (1.0 + t_mag_sq)
        s_y = 2.0 * t_y / (1.0 + t_mag_sq)
        s_z = 2.0 * t_z / (1.0 + t_mag_sq)

        # Cross product 1: v_prime = v_minus + (v_minus x t)
        v_prime_x = v_minus_x + (v_minus_y * t_z - v_minus_z * t_y)
        v_prime_y = v_minus_y + (v_minus_z * t_x - v_minus_x * t_z)
        v_prime_z = v_minus_z + (v_minus_x * t_y - v_minus_y * t_x)

        # Cross product 2: v_plus = v_minus + (v_prime x s)
        v_plus_x = v_minus_x + (v_prime_y * s_z - v_prime_z * s_y)
        v_plus_y = v_minus_y + (v_prime_z * s_x - v_prime_x * s_z)
        v_plus_z = v_minus_z + (v_prime_x * s_y - v_prime_y * s_x)

        # STEP 3: Second half E-field acceleration
        vx = v_plus_x + (q_m * Ex_p * dt) / 2.0
        vy = v_plus_y + (q_m * Ey_p * dt) / 2.0
        vz = v_plus_z # Ez is 0

        # KINEMATIC UPDATE (Notice no Z update, matching your 2D3V layout)
        px += vx * dt * 1000.0
        py += vy * dt * 1000.0
        
        # =====================================================================
        
        traj_num_x.append(px)
        traj_num_y.append(py)

    # 4. Analytical Trajectory (Mapped to mm)
    t_pts = np.linspace(0, steps*dt, steps)
    # The particle starts at (0,0) and moves in +X. The Lorentz force pushes it in -Y.
    x_exact = r_analytical_mm * np.sin(omega * t_pts)
    y_exact = r_analytical_mm * (np.cos(omega * t_pts) - 1)

    # 5. Plotting
    plt.figure(figsize=(8, 8))
    plt.plot(traj_num_x, traj_num_y, 'r-', linewidth=2, label="Engine (Taichi Logic)")
    plt.plot(x_exact, y_exact, 'k--', alpha=0.7, label="Analytical (True Circle)")
    
    plt.axis('equal')
    
    # Calculate radius error at the end of the simulation
    final_radius_num = np.sqrt(traj_num_x[-1]**2 + (traj_num_y[-1] + r_analytical_mm)**2)
    radius_error_percent = abs(final_radius_num - r_analytical_mm) / r_analytical_mm * 100
    
    plt.title(f"DigitalTwinSimulator Boris Pusher Test\nRadius: {r_analytical_mm:.2f} mm | Error: {radius_error_percent:.4f}%")
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_digital_twin_boris()