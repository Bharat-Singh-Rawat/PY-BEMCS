import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
import matplotlib.pyplot as plt

def test_digital_twin_poisson():
    # 1. Setup Domain (Units: mm)
    L = 20.0  
    dx_list = [0.25, 0.125,0.1,0.05,0.01] 
    errors = []
    eps0 = 8.854e-12

    print("Running Poisson Convergence Test ...")

    for dx in dx_list:
        dy = dx
        nx = int(L / dx) + 1
        ny = int(L / dy) + 1
        N = nx * ny
        
        x_pts = np.linspace(0, L, nx)
        y_pts = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x_pts, y_pts)

        # 2. Analytical Solution: V = sin(pi * x / L) * sin(pi * y / L)
        k = np.pi / L
        V_exact = np.sin(k * X) * np.sin(k * Y)
        
        # 3. Source Term (Rho) - Convert k to 1/m for physics
        k_m = k * 1000.0 
        rho = eps0 * (2 * k_m**2) * V_exact

        # =====================================================================
        # EXACT REPLICA OF YOUR: build_sparse_matrix()
        # =====================================================================
        isBound = np.zeros((ny, nx), dtype=bool)
        
        # For this test, we force all edges to be Dirichlet (V=0) 
        # to match our sine wave which is 0 at the boundaries.
        isBound[0, :] = True
        isBound[-1, :] = True
        isBound[:, 0] = True
        isBound[:, -1] = True

        idx = np.arange(N)
        y = idx // nx
        x = idx % nx

        is_bound = isBound.flatten()
        is_right = (x == nx - 1) & ~is_bound
        is_top = (y == ny - 1) & ~is_bound & ~is_right
        is_bottom = (y == 0) & ~is_bound & ~is_right & ~is_top
        is_interior = ~is_bound & ~is_right & ~is_top & ~is_bottom

        row, col, data = [], [], []

        idx_b = idx[is_bound]
        row.append(idx_b); col.append(idx_b); data.append(np.ones_like(idx_b))

        idx_r = idx[is_right]
        row.append(idx_r); col.append(idx_r); data.append(np.ones_like(idx_r))
        row.append(idx_r); col.append(idx_r - 1); data.append(-np.ones_like(idx_r))

        idx_t = idx[is_top]
        row.append(idx_t); col.append(idx_t); data.append(np.ones_like(idx_t))
        row.append(idx_t); col.append(idx_t - nx); data.append(-np.ones_like(idx_t))

        idx_bot = idx[is_bottom]
        row.append(idx_bot); col.append(idx_bot); data.append(np.ones_like(idx_bot))
        row.append(idx_bot); col.append(idx_bot + nx); data.append(-np.ones_like(idx_bot))

        idx_in = idx[is_interior]
        row.append(idx_in); col.append(idx_in); data.append(np.full_like(idx_in, -4.0))
        row.append(idx_in); col.append(idx_in - 1); data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in + 1); data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in - nx); data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in + nx); data.append(np.ones_like(idx_in))

        row = np.concatenate(row)
        col = np.concatenate(col)
        data = np.concatenate(data)
        
        A = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsc()
        laplacian_lu = factorized(A)

        # =====================================================================
        # recalc_poisson()
        # =====================================================================
        dx_m2 = (dx * 1e-3)**2 
        coeff = dx_m2 / eps0
        
        b = np.zeros(N, dtype=np.float32)
        V_fixed_flat = np.zeros(N, dtype=np.float32) # V is 0 at bounds for this test
        rho_flat = rho.flatten()

        b.fill(0.0)
        b[is_bound] = V_fixed_flat[is_bound]
        
        # NOTE: Your exact logic with the negative sign!
        b[is_interior] = -coeff * rho_flat[is_interior] 

        V_new_flat = laplacian_lu(b)
        V_num = V_new_flat.reshape((ny, nx))

        # =====================================================================
        # 4. Error Calculation
        # =====================================================================
        error = np.sqrt(np.mean((V_num - V_exact)**2))
        errors.append(error)
        print(f"dx: {dx:.3f} mm | Grid: {nx}x{ny} | RMS Error: {error:.2e}")

    # --- Plotting Results ---
    plt.figure(figsize=(10, 6))
    
    plt.loglog(dx_list, errors, 'ro-', linewidth=2, label="Measured Engine Error")
    
    # Calculate expected O(dx^2) ideal slope line
    expected_error = [errors[-1] * (d / dx_list[-1])**2 for d in dx_list]
    plt.loglog(dx_list, expected_error, 'k--', alpha=0.6, label=r"Ideal $O(\Delta x^2)$ Slope")

    plt.gca().invert_xaxis() 
    plt.title("DigitalTwinSimulator Poisson Accuracy")
    plt.xlabel(r"Grid Spacing $\Delta x$ (mm)")
    plt.ylabel("Root Mean Square Error")
    
    slope = np.polyfit(np.log(dx_list), np.log(errors), 1)[0]
    plt.text(dx_list[1], errors[1], f"Measured Slope: {slope:.2f}\n(A slope of ~2.0 is mathematically perfect)", 
             bbox=dict(facecolor='white', alpha=0.9))
    
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_digital_twin_poisson()