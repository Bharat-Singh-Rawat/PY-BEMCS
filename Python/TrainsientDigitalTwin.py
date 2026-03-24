import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton, 
                             QCheckBox, QFrame, QMessageBox, QFileDialog)
from PyQt5.QtCore import QTimer, Qt

class DigitalTwinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('2-Grid Digital Twin: Morphing, Telemetry & LIVE 3D')
        self.setGeometry(20, 30, 1280, 780)

        # --- 1. GLOBAL STATE & SIMULATION VARIABLES ---
        self.sim_isRunning = False
        self.dt = 2e-9
        self.q = 1.602e-19
        self.m_XE = 131.293 * 1.6605e-27
        self.kB = 1.380649e-23
        
        # Mesh parameters
        self.Lx = 8.0
        self.Ly = 4.0
        self.dx = 0.05
        self.dy = 0.05
        self.nx = int(self.Lx / self.dx) + 1
        self.ny = int(self.Ly / self.dy) + 1
        
        self.x_pts = np.linspace(0, self.Lx, self.nx)
        self.y_pts = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x_pts, self.y_pts)
        
        self.reset_arrays()

        # Telemetry
        self.iteration = 0
        self.iter_history = []
        self.ebs_history = []
        self.div_history = []
        self.recorded_frames = []

        # Setup GUI
        self.setup_ui()
        
        # Setup Timer for Main Loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.sim_step)
        self.timer.start(10) # 10ms loop

        # 3D references
        self.fig3d = None
        self.ax3d = None

    def reset_arrays(self):
        self.p_x = np.array([])
        self.p_y = np.array([])
        self.p_vx = np.array([])
        self.p_vy = np.array([])
        self.p_isCEX = np.array([], dtype=bool)
        
        self.V = np.zeros((self.ny, self.nx))
        self.isBound = np.zeros((self.ny, self.nx), dtype=bool)
        self.V_fixed = np.zeros((self.ny, self.nx))
        self.damage_map = np.zeros((self.ny, self.nx))
        self.Ex = np.zeros((self.ny, self.nx))
        self.Ey = np.zeros((self.ny, self.nx))

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- LEFT PANEL (Controls) ---
        control_panel = QFrame()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)

        # Helper to create inputs
        self.inputs = {}
        def add_input(label_text, default_val, min_v, max_v, step, decimals=1):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(130)
            spin = QDoubleSpinBox()
            spin.setRange(min_v, max_v)
            spin.setValue(default_val)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            row.addWidget(lbl)
            row.addWidget(spin)
            control_layout.addLayout(row)
            return spin

        lbl_grid = QLabel('<b>1. GRID DESIGN</b>')
        lbl_grid.setStyleSheet("color: #1A4D99;")
        control_layout.addWidget(lbl_grid)
        
        self.inputs['Vs'] = add_input('Screen Volts (V):', 1000, 0, 5000, 100, 0)
        self.inputs['Va'] = add_input('Accel Volts (V):', -200, -1000, 0, 50, 0)
        self.inputs['gap'] = add_input('Grid Gap (mm):', 1.0, 0.1, 5.0, 0.1)
        self.inputs['ts'] = add_input('Screen Thick (mm):', 0.6, 0.1, 5.0, 0.1)
        self.inputs['ta'] = add_input('Accel Thick (mm):', 1.2, 0.1, 5.0, 0.1)
        self.inputs['rs'] = add_input('Screen Hole R (mm):', 1.0, 0.1, 5.0, 0.1)
        self.inputs['ra'] = add_input('Accel Hole R (mm):', 0.8, 0.1, 5.0, 0.1)
        self.inputs['cham_s'] = add_input('Screen Chamfer (°):', 0, 0, 45, 1)
        self.inputs['cham_a'] = add_input('Accel Chamfer (°):', 15, 0, 45, 1)

        control_layout.addSpacing(15)
        lbl_plasma = QLabel('<b>2. PLASMA & MORPHING</b>')
        lbl_plasma.setStyleSheet("color: #1A4D99;")
        control_layout.addWidget(lbl_plasma)

        self.inputs['Ti'] = add_input('Ion Temp (eV):', 2.0, 0.1, 10, 0.5)
        self.inputs['Tn'] = add_input('Neutral Temp (K):', 300, 100, 2000, 100, 0)
        self.inputs['n0'] = add_input('Neutral Dens n0:', 1e20, 1e18, 1e22, 1e19, 0)
        self.inputs['Accel'] = add_input('Accel. Factor (X):', 5e13, 1e10, 1e16, 1e12, 0)
        self.inputs['Thresh'] = add_input('Cell Fail Thresh:', 1.0, 0.1, 10.0, 0.1)

        control_layout.addSpacing(15)

        self.btn_build = QPushButton('1. BUILD DOMAIN')
        self.btn_build.setStyleSheet("background-color: #FFEA99; font-weight: bold;")
        self.btn_build.clicked.connect(self.build_domain)
        control_layout.addWidget(self.btn_build)

        self.btn_toggle = QPushButton('2. START BEAM EXTRACTION')
        self.btn_toggle.setStyleSheet("background-color: #CCFFCC; font-weight: bold;")
        self.btn_toggle.clicked.connect(self.toggle_sim)
        control_layout.addWidget(self.btn_toggle)

        self.btn_3d = QPushButton('OPEN LIVE 3D CAD VIEW')
        self.btn_3d.setStyleSheet("background-color: #E0F7FA; font-weight: bold;")
        self.btn_3d.clicked.connect(self.init_3d)
        control_layout.addWidget(self.btn_3d)

        self.chk_record = QCheckBox('Record Frames (0)')
        control_layout.addWidget(self.chk_record)

        self.btn_save = QPushButton('Save GIF Animation')
        self.btn_save.clicked.connect(self.save_gif)
        control_layout.addWidget(self.btn_save)

        self.lbl_status = QLabel('Status: Ready.')
        self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
        control_layout.addWidget(self.lbl_status)

        main_layout.addWidget(control_panel)

        # --- RIGHT PANEL (Matplotlib Canvas) ---
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.patch.set_facecolor('#F4F6F9')
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # Create subplots
        grid = plt.GridSpec(2, 3, height_ratios=[1.2, 1])
        
        self.ax_live = self.fig.add_subplot(grid[0, :])
        self.ax_live.set_title('Live Axisymmetric Plasma Extraction & E-Field')
        self.ax_live.set_aspect('equal')
        self.ax_live.set_xlim(0, self.Lx)
        self.ax_live.set_ylim(0, self.Ly)

        self.ax_dmg = self.fig.add_subplot(grid[1, 0])
        self.ax_dmg.set_title('Sputter Damage Map')
        
        self.ax_ebs = self.fig.add_subplot(grid[1, 1])
        self.ax_ebs.set_title('Electron Backstreaming')
        self.ax_ebs.grid(True)
        self.line_ebs, = self.ax_ebs.plot([], [], 'm-', lw=2)
        self.ax_ebs.axhline(-20, color='r', linestyle='--', label='Fail Limit')
        self.ax_ebs.legend(loc='lower left')

        self.ax_div = self.fig.add_subplot(grid[1, 2])
        self.ax_div.set_title('Primary Beam Divergence')
        self.ax_div.grid(True)
        self.line_div, = self.ax_div.plot([], [], 'b-', lw=2)

        self.fig.tight_layout()

        # Scatter plot handles
        self.scat_prim = self.ax_live.scatter([], [], s=2, c='b', alpha=0.6)
        self.scat_cex = self.ax_live.scatter([], [], s=5, c='r')
        self.scat_bound = None

    def toggle_sim(self):
        if not np.any(self.Ex):
            QMessageBox.warning(self, "Warning", "Build Domain first!")
            return
        self.sim_isRunning = not self.sim_isRunning
        if self.sim_isRunning:
            self.btn_toggle.setText('2. PAUSE BEAM EXTRACTION')
            self.btn_toggle.setStyleSheet("background-color: #FFCCCC; font-weight: bold;")
        else:
            self.btn_toggle.setText('2. RESUME BEAM EXTRACTION')
            self.btn_toggle.setStyleSheet("background-color: #CCFFCC; font-weight: bold;")

    def build_domain(self):
        self.sim_isRunning = False
        self.btn_toggle.setText('2. START BEAM EXTRACTION')
        self.btn_toggle.setStyleSheet("background-color: #CCFFCC; font-weight: bold;")
        
        self.reset_arrays()
        self.iteration = 0
        self.iter_history.clear()
        self.ebs_history.clear()
        self.div_history.clear()

        self.lbl_status.setText('Building 2-Grid Domain...')
        QApplication.processEvents()

        Vs = self.inputs['Vs'].value()
        Va = self.inputs['Va'].value()
        ts = self.inputs['ts'].value()
        ta = self.inputs['ta'].value()
        gap = self.inputs['gap'].value()
        rs = self.inputs['rs'].value()
        ra = self.inputs['ra'].value()
        cham_s = self.inputs['cham_s'].value()
        cham_a = self.inputs['cham_a'].value()

        screen_start = 1.0
        screen_end = screen_start + ts
        accel_start = screen_end + gap
        accel_end = accel_start + ta

        in_screen = (self.X >= screen_start) & (self.X <= screen_end)
        in_accel = (self.X >= accel_start) & (self.X <= accel_end)

        R_screen = rs + np.maximum(0, self.X - screen_start) * np.tan(np.radians(cham_s))
        R_accel = ra + np.maximum(0, self.X - accel_start) * np.tan(np.radians(cham_a))

        mask_s = in_screen & (self.Y >= R_screen)
        self.isBound[mask_s] = True
        self.V_fixed[mask_s] = Vs

        mask_a = in_accel & (self.Y >= R_accel)
        self.isBound[mask_a] = True
        self.V_fixed[mask_a] = Va

        # Left Boundary
        self.isBound[:, 0] = True
        self.V_fixed[:, 0] = Vs + 50

        self.recalc_laplace()
        self.lbl_status.setText('Domain Ready.')

    def recalc_laplace(self):
        self.V[self.isBound] = self.V_fixed[self.isBound]
        
        # 500 Iterations of Finite Difference
        for _ in range(500):
            self.V[1:-1, 1:-1] = 0.25 * (self.V[2:, 1:-1] + self.V[:-2, 1:-1] + self.V[1:-1, 2:] + self.V[1:-1, :-2])
            self.V[self.isBound] = self.V_fixed[self.isBound]
            self.V[0, :] = self.V[1, :]
            self.V[-1, :] = self.V[-2, :]
            self.V[:, -1] = self.V[:, -2]

        self.Ey, self.Ex = np.gradient(-self.V, self.dy * 1e-3, self.dx * 1e-3)
        self.interp_Ex = RegularGridInterpolator((self.y_pts, self.x_pts), self.Ex, bounds_error=False, fill_value=0)
        self.interp_Ey = RegularGridInterpolator((self.y_pts, self.x_pts), self.Ey, bounds_error=False, fill_value=0)

        # Redraw Live Axis
        self.ax_live.clear()
        self.ax_live.set_title('Live Axisymmetric Plasma Extraction & E-Field')
        self.ax_live.contourf(self.X, self.Y, self.V, 20, cmap='turbo')
        
        gy, gx = np.where(self.isBound)
        self.scat_bound = self.ax_live.scatter(gx * self.dx, gy * self.dy, s=12, c='k', alpha=0.8)
        self.scat_prim = self.ax_live.scatter([], [], s=2, c='b', alpha=0.6)
        self.scat_cex = self.ax_live.scatter([], [], s=5, c='r')
        
        self.canvas.draw()

    def sim_step(self):
        if not self.sim_isRunning or not np.any(self.Ex):
            return

        self.iteration += 1

        # 1. Inject Primaries
        num_inject = 30
        rs_val = self.inputs['rs'].value()
        new_y = np.linspace(0.05, rs_val - 0.1, num_inject) + (np.random.rand(num_inject) - 0.5) * 0.05
        new_x = np.full(num_inject, 0.1)

        v_bohm = np.sqrt(2 * self.q * 50 / self.m_XE)
        v_spread = np.sqrt(self.q * self.inputs['Ti'].value() / self.m_XE)

        self.p_x = np.concatenate((self.p_x, new_x))
        self.p_y = np.concatenate((self.p_y, new_y))
        self.p_vx = np.concatenate((self.p_vx, np.full(num_inject, v_bohm) + np.random.randn(num_inject) * v_spread))
        self.p_vy = np.concatenate((self.p_vy, np.random.randn(num_inject) * v_spread))
        self.p_isCEX = np.concatenate((self.p_isCEX, np.zeros(num_inject, dtype=bool)))

        # 2. Push Particles
        pts = np.column_stack((self.p_y, self.p_x))
        Ex_p = self.interp_Ex(pts)
        Ey_p = self.interp_Ey(pts)

        self.p_vx += (self.q / self.m_XE) * Ex_p * self.dt
        self.p_vy += (self.q / self.m_XE) * Ey_p * self.dt
        self.p_x += self.p_vx * self.dt * 1000
        self.p_y += self.p_vy * self.dt * 1000

        # 3. Telemetry
        max_grid_x = 1.0 + self.inputs['ts'].value() + self.inputs['gap'].value() + self.inputs['ta'].value()
        post_grid = (~self.p_isCEX) & (self.p_x > max_grid_x)
        
        if np.sum(post_grid) > 5:
            angles = np.abs(np.arctan2(self.p_vy[post_grid], self.p_vx[post_grid])) * 180 / np.pi
            current_div = np.percentile(angles, 95)
        else:
            current_div = np.nan

        min_pot = np.min(self.V[0, :])

        # 4. Hit Detection
        ix = np.clip(np.round(self.p_x / self.dx).astype(int), 0, self.nx - 1)
        iy = np.clip(np.round(self.p_y / self.dy).astype(int), 0, self.ny - 1)
        
        hit_grid = self.isBound[iy, ix]
        out_of_bounds = (self.p_x < 0) | (self.p_x > self.Lx) | (self.p_y < 0) | (self.p_y > self.Ly) | np.isnan(self.p_x)

        # 5. Sputtering & Morphing
        is_erosion_hit = hit_grid & self.p_isCEX
        if np.any(is_erosion_hit):
            v_mag_sq = self.p_vx[is_erosion_hit]**2 + self.p_vy[is_erosion_hit]**2
            E_eV = (0.5 * self.m_XE * v_mag_sq) / self.q
            
            Y_yield = np.zeros_like(E_eV)
            valid_E = E_eV > 30
            Y_yield[valid_E] = 1.05e-4 * (E_eV[valid_E] - 30)**1.5

            hit_iy = iy[is_erosion_hit]
            hit_ix = ix[is_erosion_hit]
            
            # Accumulate damage
            np.add.at(self.damage_map, (hit_iy, hit_ix), Y_yield * self.inputs['Accel'].value())

            # Structural failure check
            broken_cells = (self.damage_map > self.inputs['Thresh'].value()) & self.isBound
            if np.any(broken_cells):
                self.isBound[broken_cells] = False
                self.damage_map[broken_cells] = 0
                self.lbl_status.setText('Cell Failed! Remeshing Laplace...')
                self.recalc_laplace()

        # Purge dead
        dead_mask = hit_grid | out_of_bounds
        self.p_x = self.p_x[~dead_mask]
        self.p_y = self.p_y[~dead_mask]
        self.p_vx = self.p_vx[~dead_mask]
        self.p_vy = self.p_vy[~dead_mask]
        self.p_isCEX = self.p_isCEX[~dead_mask]

        # 6. Charge Exchange
        primary_mask = (~self.p_isCEX) & (self.p_x > 1.0)
        if np.any(primary_mask):
            v_mag = np.sqrt(self.p_vx[primary_mask]**2 + self.p_vy[primary_mask]**2)
            g = np.maximum(v_mag, 1)
            sigma = ((-0.8821 * np.log(g) + 15.1262)**2) * 1e-20
            prob = 1 - np.exp(-self.inputs['n0'].value() * sigma * g * self.dt)
            
            collided = np.random.rand(np.sum(primary_mask)) < (prob * 1) # Speed factor kept identical to MATLAB
            
            if np.any(collided):
                idx = np.where(primary_mask)[0]
                c_idx = idx[collided]
                num_cex = len(c_idx)
                neut_vth = np.sqrt(2 * self.kB * self.inputs['Tn'].value() / self.m_XE)
                
                self.p_vx[c_idx] = np.random.randn(num_cex) * neut_vth
                self.p_vy[c_idx] = np.random.randn(num_cex) * neut_vth
                self.p_isCEX[c_idx] = True

        # 7. UI Update
        if self.iteration % 5 == 0:
            prim_mask = ~self.p_isCEX
            self.scat_prim.set_offsets(np.column_stack((self.p_x[prim_mask], self.p_y[prim_mask])))
            self.scat_cex.set_offsets(np.column_stack((self.p_x[self.p_isCEX], self.p_y[self.p_isCEX])))

            self.iter_history.append(self.iteration)
            self.ebs_history.append(min_pot)
            self.div_history.append(current_div)

            self.line_ebs.set_data(self.iter_history, self.ebs_history)
            self.line_div.set_data(self.iter_history, self.div_history)
            
            self.ax_ebs.set_xlim(max(0, self.iteration - 400), max(100, self.iteration))
            self.ax_ebs.set_ylim(min(-30, min(self.ebs_history)), max(10, max(self.ebs_history)))
            self.ax_div.set_xlim(max(0, self.iteration - 400), max(100, self.iteration))
            self.ax_div.set_ylim(0, 45)

            # Update Damage map
            self.ax_dmg.clear()
            self.ax_dmg.set_title('Sputter Damage Map')
            self.ax_dmg.contourf(self.X, self.Y, self.damage_map, 15, cmap='hot')
            self.ax_dmg.set_xlim(0.5, self.Lx)
            self.ax_dmg.set_ylim(0, self.Ly)
            gy, gx = np.where(self.isBound)
            self.ax_dmg.scatter(gx * self.dx, gy * self.dy, s=2, c='grey', alpha=0.5)

            self.lbl_status.setText(f'Active Particles: {len(self.p_x)} | Iteration: {self.iteration}')
            self.canvas.draw()

            if self.chk_record.isChecked():
                img = self.canvas.grab().toImage()
                self.recorded_frames.append(img)
                self.chk_record.setText(f'Record Frames ({len(self.recorded_frames)})')

    def init_3d(self):
        if not np.any(self.isBound):
            QMessageBox.warning(self, "Warning", "Build domain first to generate 3D model.")
            return

        if self.fig3d is None or not plt.fignum_exists(self.fig3d.number):
            self.fig3d = plt.figure(figsize=(9, 7))
            self.fig3d.patch.set_facecolor('black')
            self.ax3d = self.fig3d.add_subplot(111, projection='3d')
            self.ax3d.set_facecolor('black')
            
            self.ax3d.xaxis.label.set_color('white')
            self.ax3d.yaxis.label.set_color('white')
            self.ax3d.zaxis.label.set_color('white')
            self.ax3d.tick_params(axis='x', colors='white')
            self.ax3d.tick_params(axis='y', colors='white')
            self.ax3d.tick_params(axis='z', colors='white')
            self.ax3d.set_title("Live 3D Structure View (Placeholder in Realtime)", color='white')
            
            # Simple Static 3D Surface mapping for Python optimization
            x_range = self.X[0, :]
            R_profile = np.zeros_like(x_range)
            for i in range(len(x_range)):
                solid_y = np.where(self.isBound[:, i])[0]
                if len(solid_y) > 0:
                    R_profile[i] = self.Y[solid_y[0], 0]
                else:
                    R_profile[i] = np.nan
            
            theta = np.linspace(0, 2*np.pi, 30)
            Theta_mat, X_mat = np.meshgrid(theta, x_range)
            R_mat = np.tile(R_profile, (len(theta), 1)).T
            
            Y_mat = R_mat * np.cos(Theta_mat)
            Z_mat = R_mat * np.sin(Theta_mat)

            self.ax3d.plot_surface(X_mat, Y_mat, Z_mat, cmap='hot', edgecolor='none')
            self.ax3d.view_init(elev=0, azim=-90)
            self.fig3d.show()

    def save_gif(self):
        if not self.recorded_frames:
            QMessageBox.warning(self, "Error", "No frames recorded!")
            return
            
        was_running = self.sim_isRunning
        self.sim_isRunning = False
        
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Animation", "", "GIF Files (*.gif)", options=options)
        
        if file_name:
            # We use Pillow to save QImages to a GIF
            try:
                from PIL import Image
                pil_images = []
                for qimg in self.recorded_frames:
                    qimg = qimg.convertToFormat(4) # Format_RGB32
                    width = qimg.width()
                    height = qimg.height()
                    ptr = qimg.bits()
                    ptr.setsize(qimg.byteCount())
                    arr = np.array(ptr).reshape(height, width, 4)
                    pil_images.append(Image.fromarray(arr[..., [2, 1, 0]], 'RGB'))
                
                pil_images[0].save(file_name, save_all=True, append_images=pil_images[1:], optimize=False, duration=50, loop=0)
                
                self.recorded_frames.clear()
                self.chk_record.setChecked(False)
                self.chk_record.setText('Record Frames (0)')
                QMessageBox.information(self, "Success", "GIF Saved Successfully!")
            except ImportError:
                QMessageBox.warning(self, "Dependency Error", "Please install Pillow (`pip install Pillow`) to save GIFs.")
                
        self.sim_isRunning = was_running

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DigitalTwinApp()
    ex.show()
    sys.exit(app.exec_())
    