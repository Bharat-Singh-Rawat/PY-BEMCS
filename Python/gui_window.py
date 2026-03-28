import sys 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QDoubleSpinBox, QPushButton, QCheckBox, 
                             QFrame, QMessageBox, QFileDialog, QApplication, QComboBox)
from PyQt5.QtCore import QTimer, Qt
import csv

# Ensure your updated physics engine is saved as physics_engine.py in the same directory
from physics_engine import DigitalTwinSimulator

class DigitalTwinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('2-Grid Digital Twin: Morphing, Telemetry & Neutralizer')
        self.setGeometry(20, 30, 1400, 800) 

        self.sim = DigitalTwinSimulator()
        self.sim_isRunning = False
        
        self.iter_history = []
        self.ebs_history = []
        self.div_history = []
        self.Ts_history = []
        self.Ta_history = []
        self.recorded_frames = []
        
        self.cbar = None

        self.setup_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_sim_step)
        self.timer.start(10)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        control_panel = QFrame()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)
        self.inputs = {}

        def add_input(label_text, default_val, min_v, max_v, step, decimals=1):
            row = QHBoxLayout()
            lbl = QLabel(label_text); lbl.setFixedWidth(130)
            spin = QDoubleSpinBox(); spin.setRange(min_v, max_v); spin.setValue(default_val)
            spin.setSingleStep(step); spin.setDecimals(decimals)
            row.addWidget(lbl); row.addWidget(spin)
            control_layout.addLayout(row)
            return spin

        control_layout.addWidget(QLabel('<b>1. GRID DESIGN</b>'))
        self.inputs['Vs'] = add_input('Screen Volts (V):', 1650, 0, 5000, 100, 0)
        self.inputs['Va'] = add_input('Accel Volts (V):', -350, -1000, 0, 50, 0)
        self.inputs['gap'] = add_input('Grid Gap (mm):', 1.0, 0.1, 5.0, 0.1)
        self.inputs['ts'] = add_input('Screen Thick (mm):', 1, 0.1, 5.0, 0.1)
        self.inputs['ta'] = add_input('Accel Thick (mm):', 1, 0.1, 5.0, 0.1)
        self.inputs['rs'] = add_input('Screen Hole R (mm):', 1.0, 0.1, 5.0, 0.1)
        self.inputs['ra'] = add_input('Accel Hole R (mm):', 0.6, 0.1, 5.0, 0.1)
        self.inputs['cham_s'] = add_input('Screen Chamfer (°):', 0, 0, 45, 1)
        self.inputs['cham_a'] = add_input('Accel Chamfer (°):', 0, 0, 45, 1)

        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel('<b>2. PLASMA & MORPHING</b>'))
        self.inputs['n0_plasma'] = add_input('Plasma Dens (m-3):', 1e17, 1e15, 1e19, 1e16, 0)
        self.inputs['Te_up'] = add_input('Upstream Te (eV):', 3.0, 0.1, 20.0, 0.5, 1)
        
        self.inputs['Ti'] = add_input('Ion Temp (eV):', 2.0, 0.1, 10, 0.5)
        self.inputs['Tn'] = add_input('Neutral Temp (K):', 300, 100, 2000, 100, 0)
        self.inputs['n0'] = add_input('Neutral Dens (m-3):', 1e20, 1e18, 1e22, 1e19, 0)
        self.inputs['Accel'] = add_input('Accel. Factor (X):', 1, 10, 1e16, 1e12, 0)
        self.inputs['Thresh'] = add_input('Cell Fail Thresh:', 1.0, 0.1, 100000.0, 0.1)

        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel('<b>3. SIMULATION MODE</b>'))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(['Both', 'Thermal', 'Erosion'])
        control_layout.addWidget(self.combo_mode)
        
        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel('<b>4. NEUTRALIZER</b>'))
        self.inputs['neut_rate'] = add_input('e- Inject Rate:', 30, 0, 200, 10, 0)
        self.inputs['Te'] = add_input('e- Temp (eV):', 5.0, 0.1, 20.0, 0.5, 1)
        control_layout.addSpacing(15)

        self.btn_build = QPushButton('1. BUILD DOMAIN')
        self.btn_build.clicked.connect(self.build_domain)
        self.btn_toggle = QPushButton('2. START BEAM')
        self.btn_toggle.clicked.connect(self.toggle_sim)
        
        self.btn_csv = QPushButton('Export Telemetry (.csv)')
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_csv.setStyleSheet("background-color: #E6E6FA; font-weight: bold;")
        
        self.chk_record = QCheckBox('Record Frames (0)')
        self.btn_save = QPushButton('Save GIF Animation')
        self.btn_save.clicked.connect(self.save_gif)
        
        self.lbl_status = QLabel('Status: Ready.')
        self.lbl_temp = QLabel('Grid Temps: Screen: 26°C | Accel: 26°C')
        self.lbl_temp.setStyleSheet("color: #D35400; font-weight: bold;") 

        for w in [self.btn_build, self.btn_toggle, self.btn_csv, self.chk_record, self.btn_save, self.lbl_status, self.lbl_temp]:
            control_layout.addWidget(w)
            
        main_layout.addWidget(control_panel)

        self.fig = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        grid = plt.GridSpec(2, 3, height_ratios=[1.2, 1])
        self.ax_live = self.fig.add_subplot(grid[0, 0:2]); self.ax_live.set_title('Live Axisymmetric Plasma')
        self.ax_temp = self.fig.add_subplot(grid[0, 2]); self.ax_temp.set_title('Grid Temp Map (°C)')
        self.ax_dmg = self.fig.add_subplot(grid[1, 0]); self.ax_dmg.set_title('Damage Map')
        self.ax_ebs = self.fig.add_subplot(grid[1, 1]); self.ax_ebs.set_title('Electron Backstreaming')
        self.ax_div = self.fig.add_subplot(grid[1, 2]); self.ax_div.set_title('Beam Divergence')
        
        self.line_ebs, = self.ax_ebs.plot([], [], 'm-', lw=2)
        self.line_div, = self.ax_div.plot([], [], 'b-', lw=2)
        
        self.scat_prim = self.ax_live.scatter([], [], s=2, c='b', alpha=0.6) 
        self.scat_cex = self.ax_live.scatter([], [], s=5, c='r')             
        self.scat_elec = self.ax_live.scatter([], [], s=1, c='#00FF00', alpha=0.5) 
        self.fig.tight_layout()

    def get_params(self):
        params = {k: v.value() for k, v in self.inputs.items()}
        params['sim_mode'] = self.combo_mode.currentText()
        return params

    def toggle_sim(self):
        if not np.any(self.sim.Ex):
            QMessageBox.warning(self, "Warning", "Build Domain first!")
            return
        self.sim_isRunning = not self.sim_isRunning
        self.btn_toggle.setText('PAUSE BEAM' if self.sim_isRunning else 'RESUME BEAM')

    def build_domain(self):
        self.sim_isRunning = False
        self.btn_toggle.setText('2. START BEAM')
        self.iter_history.clear(); self.ebs_history.clear(); self.div_history.clear()
        self.Ts_history.clear(); self.Ta_history.clear()
        
        self.lbl_status.setText('Building 2-Grid Domain...')
        QApplication.processEvents() 
        
        self.sim.build_domain(self.get_params())
        self.draw_static_domain()
        
        # ---> THIS LINE WAS UPDATED TO SHOW Wt:
        self.lbl_status.setText(f'Domain Ready. Wt: {self.sim.macro_weight:.1e}')
        self.lbl_temp.setText('Grid Temps: Screen: 26°C | Accel: 26°C')

    def draw_static_domain(self):
        self.ax_live.clear()
        self.ax_live.contourf(self.sim.X, self.sim.Y, self.sim.V, 20, cmap='turbo')
        gy, gx = np.where(self.sim.isBound)
        self.ax_live.scatter(gx * self.sim.dx, gy * self.sim.dy, s=12, c='k', alpha=0.8)
        
        self.scat_prim = self.ax_live.scatter([], [], s=2, c='b', alpha=0.6)
        self.scat_cex = self.ax_live.scatter([], [], s=5, c='r')
        self.scat_elec = self.ax_live.scatter([], [], s=1, c='#00FF00', alpha=0.5)
        self.canvas.draw()

    def run_sim_step(self):
        if not self.sim_isRunning: return

        remeshed, min_pot, current_div, T_s, T_a = self.sim.step(self.get_params())
        
        if remeshed:
            self.lbl_status.setText('Domain Remeshed (Thermal or Erosion)!')
            self.draw_static_domain()

        if self.sim.iteration % 5 == 0:
            prim_mask = ~self.sim.p_isCEX
            self.scat_prim.set_offsets(np.column_stack((self.sim.p_x[prim_mask], self.sim.p_y[prim_mask])) if len(self.sim.p_x[prim_mask]) > 0 else np.empty((0, 2)))
            self.scat_cex.set_offsets(np.column_stack((self.sim.p_x[self.sim.p_isCEX], self.sim.p_y[self.sim.p_isCEX])) if len(self.sim.p_x[self.sim.p_isCEX]) > 0 else np.empty((0, 2)))
            self.scat_elec.set_offsets(np.column_stack((self.sim.e_x, self.sim.e_y)) if len(self.sim.e_x) > 0 else np.empty((0, 2)))

            self.iter_history.append(self.sim.iteration)
            self.ebs_history.append(min_pot)
            self.div_history.append(current_div)
            self.Ts_history.append(T_s)
            self.Ta_history.append(T_a)

            self.line_ebs.set_data(self.iter_history, self.ebs_history)
            self.line_div.set_data(self.iter_history, self.div_history)
            
            self.ax_ebs.set_xlim(max(0, self.sim.iteration - 400), max(100, self.sim.iteration))
            self.ax_ebs.set_ylim(min(-30, min(self.ebs_history)), max(10, max(self.ebs_history)))
            self.ax_div.set_xlim(max(0, self.sim.iteration - 400), max(100, self.sim.iteration))
            self.ax_div.set_ylim(0, 45)

            if self.cbar is not None:
                self.cbar.remove()
                self.cbar = None

            self.ax_temp.clear()
            self.ax_temp.set_title('Grid Temp Map (°C)')
            self.ax_temp.set_facecolor('black')
            
            if np.any(self.sim.isBound):
                T_display_C = np.copy(self.sim.T_map) - 273.15
                T_display_C[~self.sim.isBound] = np.nan 
                
                contour = self.ax_temp.contourf(self.sim.X, self.sim.Y, T_display_C, 15, cmap='inferno')
                
                self.cbar = self.fig.colorbar(contour, ax=self.ax_temp, fraction=0.046, pad=0.04)
                self.cbar.set_label('Temperature (°C)')
                
                ts = self.inputs['ts'].value()
                gap = self.inputs['gap'].value()
                ta = self.inputs['ta'].value()
                self.ax_temp.set_xlim(0.8, 1.2 + ts + gap + ta + 0.2)

            self.ax_dmg.clear()
            self.ax_dmg.set_title('Damage Map')
            self.ax_dmg.contourf(self.sim.X, self.sim.Y, self.sim.damage_map, 15, cmap='hot')
            gy, gx = np.where(self.sim.isBound)
            self.ax_dmg.scatter(gx * self.sim.dx, gy * self.sim.dy, s=2, c='grey', alpha=0.5)

            # ---> THIS LINE WAS UPDATED TO SHOW Wt:
            self.lbl_status.setText(f'Ions: {len(self.sim.p_x)} | e-: {len(self.sim.e_x)} | Iter: {self.sim.iteration} | Wt: {self.sim.macro_weight:.1e}')
            self.lbl_temp.setText(f'Grid Temps: Screen: {int(T_s - 273.15)}°C | Accel: {int(T_a - 273.15)}°C')
            self.canvas.draw()

            if self.chk_record.isChecked():
                self.recorded_frames.append(self.canvas.grab().toImage())
                self.chk_record.setText(f'Record Frames ({len(self.recorded_frames)})')

    def export_csv(self):
        if not self.iter_history:
            QMessageBox.warning(self, "Error", "No telemetry data to export! Run the simulation first.")
            return
        
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Telemetry", "", "CSV Files (*.csv)")
        if file_name:
            try:
                with open(file_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Iteration', 'Min_Potential_V', 'Divergence_Deg', 'Screen_Temp_C', 'Accel_Temp_C'])
                    
                    for i in range(len(self.iter_history)):
                        writer.writerow([
                            self.iter_history[i], 
                            round(self.ebs_history[i], 2), 
                            round(self.div_history[i], 2) if not np.isnan(self.div_history[i]) else "", 
                            round(self.Ts_history[i] - 273.15, 1), 
                            round(self.Ta_history[i] - 273.15, 1)  
                        ])
                QMessageBox.information(self, "Success", f"Telemetry Exported to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to write file:\n{str(e)}")

    def save_gif(self):
        if not self.recorded_frames:
            QMessageBox.warning(self, "Error", "No frames recorded!")
            return
        
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Animation", "", "GIF Files (*.gif)")
        if file_name:
            try:
                from PIL import Image
                pil_images = []
                for qimg in self.recorded_frames:
                    qimg = qimg.convertToFormat(4) 
                    ptr = qimg.bits(); ptr.setsize(qimg.byteCount())
                    arr = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)
                    pil_images.append(Image.fromarray(arr[..., [2, 1, 0]], 'RGB'))
                
                pil_images[0].save(file_name, save_all=True, append_images=pil_images[1:], duration=50, loop=0)
                self.recorded_frames.clear(); self.chk_record.setChecked(False)
                QMessageBox.information(self, "Success", "GIF Saved Successfully!")
            except ImportError:
                QMessageBox.warning(self, "Error", "Install Pillow (`pip install Pillow`)")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # Gives it a clean, modern look
    window = DigitalTwinApp()
    window.show()
    sys.exit(app.exec_())