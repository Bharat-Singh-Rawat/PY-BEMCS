import numpy as np
import matplotlib
# matplotlib.use('TkAgg') # Uncomment if window stays white
import matplotlib.pyplot as plt
import multiprocessing as mp
import queue
from matplotlib.ticker import PercentFormatter
from physics_engine import DigitalTwinSimulator 

# --- WORKER PROCESS FUNCTION ---
def worker_sweep(gap, n0_sweep, result_queue):
    q = 1.602e-19
    m_XE = 131.293 * 1.6605e-27
    
    for n0 in n0_sweep:
        sim = DigitalTwinSimulator()
        params = {
            'sim_mode': 'Both',
            'ts': 1, 'ta': 1, 'rs': 1.0, 'ra': 0.6,
            'cham_s': 0, 'cham_a': 0,
            'Vs': 1650, 'Va': -350, 
            'gap': gap,
            'n0_plasma': n0,
            'Te_up': 3.0, 'Ti': 0.1,
            'neut_rate': 0, 'n0': 0 
        }
        
        # NEW: Construct the expected grids array for the physics engine
        params['grids'] = [
            {'V': params['Vs'], 't': params['ts'], 'gap': params['gap'], 'r': params['rs'], 'cham': params['cham_s']},
            {'V': params['Va'], 't': params['ta'], 'gap': 1.0, 'r': params['ra'], 'cham': params['cham_a']}
        ]
        
        sim.build_domain(params)
        steady_state_steps = 500
        
        # Accumulators for the steady-state period
        total_hits = 0
        total_exits = 0
        
        for step_idx in range(1, steady_state_steps + 1):
            
            # --- OUTSIDE-IN PREDICTOR (No engine modifications needed) ---
            # We look at the arrays BEFORE sim.step() deletes the dead particles
            if len(sim.p_x) > 0:
                # Predict next positions using current velocity (1000 multiplier converts m/s to mm/s)
                next_x = sim.p_x + sim.p_vx * sim.dt * 1000
                next_y = sim.p_y + sim.p_vy * sim.dt * 1000
                
                # 1. Count particles about to exit the right boundary
                step_exits = np.sum(next_x >= sim.Lx)
                
                # 2. Count particles about to hit the accelerator grid mask
                ix = np.clip(np.round(next_x / sim.dx).astype(int), 0, sim.nx - 1)
                iy = np.clip(np.round(next_y / sim.dy).astype(int), 0, sim.ny - 1)
                
                # Filter for primary ions hitting the accelerator grid
                step_hits = np.sum(sim.mask_grids[1][iy, ix] & ~sim.p_isCEX)
            else:
                step_exits = 0
                step_hits = 0
            # -------------------------------------------------------------
            
            # Now run the actual engine step (which will internally purge them)
            sim.step(params)
            
            # Only accumulate counts during steady-state (ignore initial plasma wave)
            if step_idx > 300:
                total_hits += step_hits
                total_exits += step_exits
            
            if step_idx % 50 == 0 or step_idx == 1:
                current_ratio = step_hits / (step_hits + step_exits) if (step_hits + step_exits) > 0 else 0
                print(f"[Core Gap {gap}mm | n0: {n0:.1e}] Step {step_idx}/{steady_state_steps} -> Inst. Scraping: {current_ratio:.2%}", flush=True)

        # Calculate final steady-state impingement fraction (Hits / Total Particles)
        total_particles = total_hits + total_exits
        impingement_fraction = total_hits / total_particles if total_particles > 0 else 1.0

        # Physics Math for Perveance X-Axis
        v_bohm = np.sqrt(q * params['Te_up'] / m_XE)
        A_inject = (params['rs'] - 0.05) * 1e-3 * 1.0
        I_ion = q * 0.61 * n0 * v_bohm * A_inject
        V_total = params['Vs'] - params['Va']
        perveance = I_ion / (V_total**1.5)
        
        result_queue.put(('data', gap, perveance, impingement_fraction))
        
    result_queue.put(('done', gap))

# --- MAIN GUI THREAD ---
def run_impingement_benchmark():
    grid_gaps = [0.5, 0.75, 1.0] 
    n0_sweep = np.linspace(1e15, 5e17, 20) 
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('PY-BEMCS: Beam Impingement Fraction vs. Perveance')
    ax.set_xlabel('Perveance (A/V^{1.5})')
    ax.set_ylabel('Impingement Fraction (Hits / Total Extracted)')
    
    # Format Y-axis to display as percentages
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    lines_sim = {}
    data_store = {gap: {'perv': [], 'ratio': []} for gap in grid_gaps}
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for i, gap in enumerate(grid_gaps):
        lines_sim[gap], = ax.plot([], [], marker='o', markersize=4, color=colors[i], label=f'Gap = {gap} mm')
        
    ax.legend()
    fig.canvas.draw()
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    workers = []
    
    for gap in grid_gaps:
        p = mp.Process(target=worker_sweep, args=(gap, n0_sweep, result_queue))
        p.start()
        workers.append(p)
        
    completed_workers = 0
    while completed_workers < len(grid_gaps):
        try:
            msg = result_queue.get(timeout=0.1)
            if msg[0] == 'done':
                completed_workers += 1
            elif msg[0] == 'data':
                _, gap, perv, ratio = msg
                data_store[gap]['perv'].append(perv)
                data_store[gap]['ratio'].append(ratio)
                
                sort_idx = np.argsort(data_store[gap]['perv'])
                lines_sim[gap].set_data(np.array(data_store[gap]['perv'])[sort_idx], 
                                        np.array(data_store[gap]['ratio'])[sort_idx])
                ax.relim()
                ax.autoscale_view()
        except queue.Empty:
            pass
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    for p in workers: p.join()
    plt.ioff() 
    plt.savefig('impingement_ratio_benchmark.png', dpi=300)
    print("Benchmark complete. Plot saved as 'impingement_ratio_benchmark.png'.")
    plt.show() 

if __name__ == "__main__":
    run_impingement_benchmark()