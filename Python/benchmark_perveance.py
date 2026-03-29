import numpy as np
import matplotlib
# Uncomment the line below if your plot window freezes or remains white:
# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import multiprocessing as mp
import queue
from physics_engine import DigitalTwinSimulator  # Adjust import to match your file structure

# --- WORKER PROCESS FUNCTION ---
def worker_sweep(gap, n0_sweep, result_queue):
    """This function runs entirely on a separate CPU core."""
    q = 1.602e-19
    m_XE = 131.293 * 1.6605e-27
    
    for n0 in n0_sweep:
        sim = DigitalTwinSimulator()
        params = {
            'sim_mode': 'Both',
            'ts': 1, 'ta': 1, 
            'rs': 1.0, 'ra': 0.6,
            'cham_s': 0, 'cham_a': 0,
            'Vs': 1650, 'Va': -500, 
            'gap': gap,
            'n0_plasma': n0,
            'Te_up': 3.0, 'Ti': 0.1,
            'neut_rate': 0, 'n0': 0 ,
            'Accel': 10.0, 
            'Thresh': 10000.0
        }
        
        sim.build_domain(params)
        steady_state_steps = 1000
        div_history = []
        
        for step_idx in range(1, steady_state_steps + 1):
            _, _, current_div, _, _ = sim.step(params)
            
            if not np.isnan(current_div):
                div_history.append(current_div)
                if len(div_history) > 50:
                    div_history.pop(0) 
            
            # --- NEW: LIVE TERMINAL TRACKING ---
            # Prints the step and divergence every 50 iterations to avoid spamming the console
            if step_idx % 50 == 0 or step_idx == 1:
                print(f"[Core Gap {gap}mm | n0: {n0:.1e}] Step {step_idx}/{steady_state_steps} -> Div: {current_div:.2f}°", flush=True)
            # -----------------------------------

            # Early Stopping Check
            if step_idx >= 500 and len(div_history) == 50:
                if np.std(div_history) < 0.1: 
                    print(f"[Core Gap {gap}mm | n0: {n0:.1e}] CONVERGED EARLY at Step {step_idx}!", flush=True)
                    break

        # Calculate actual injected current for perveance math
        v_bohm = np.sqrt(q * params['Te_up'] / m_XE)
        A_inject = (params['rs'] - 0.05) * 1e-3 * 1.0
        I_ion = q * 0.61 * n0 * v_bohm * A_inject
        V_total = params['Vs'] - params['Va']
        perveance = I_ion / (V_total**1.5)
        
        final_div = np.mean(div_history) if len(div_history) > 0 else current_div
        
        # Send the finalized data point back to the main thread
        result_queue.put(('data', gap, n0, perveance, final_div))
        
    # Signal that this specific CPU core is finished
    result_queue.put(('done', gap))

# --- MAIN GUI THREAD ---
def run_parallel_benchmark():
    grid_gaps = [0.5, 0.75, 1.0] # mm
    n0_sweep = np.linspace(0.4e16, 5e17, 60) 
    
    # 1. Setup Live Plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('PY-BEMCS: Xenon Beam Divergence vs. Perveance (Without Neutralizer)')
    ax.set_xlabel('Perveance (A/V^1.5)')
    ax.set_ylabel('Divergence Angle (Degrees)')
    ax.grid(True)
    
    lines_sim = {}
    data_store = {gap: {'perveance': [], 'divergence': []} for gap in grid_gaps}
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for i, gap in enumerate(grid_gaps):
        lines_sim[gap], = ax.plot([], [], marker='o', color=colors[i], label=f'Sim (Gap = {gap} mm)')
        
    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events() 
    
    # 2. Initialize Multiprocessing Queue and Workers
    manager = mp.Manager()
    result_queue = manager.Queue()
    workers = []
    
    print("Starting Multicore Physics Engine Benchmark...")
    print("Dispatching 3 grid gaps to separate CPU cores. Monitoring steps...\n")
    print("-" * 60)
    
    for gap in grid_gaps:
        p = mp.Process(target=worker_sweep, args=(gap, n0_sweep, result_queue))
        p.start()
        workers.append(p)
        
    # 3. The Event Loop (Listen to workers and update GUI)
    completed_workers = 0
    while completed_workers < len(grid_gaps):
        try:
            msg = result_queue.get(timeout=0.1)
            
            if msg[0] == 'done':
                completed_workers += 1
                print(f"\n--> CORE FINISHED: Gap {msg[1]} mm sweep complete.\n")
                
            elif msg[0] == 'data':
                _, gap, n0, perv, div = msg
                print(f"\n*** PLOTTING POINT: [Gap {gap} mm] Perv: {perv:.2e} | Final Div: {div:.2f}° ***\n")
                
                # Append data
                data_store[gap]['perveance'].append(perv)
                data_store[gap]['divergence'].append(div)
                
                # Sort data by perveance so the line draws cleanly from left to right
                sort_idx = np.argsort(data_store[gap]['perveance'])
                sorted_perv = np.array(data_store[gap]['perveance'])[sort_idx]
                sorted_div = np.array(data_store[gap]['divergence'])[sort_idx]
                
                # Update Plot Lines
                lines_sim[gap].set_data(sorted_perv, sorted_div)
                
                ax.relim()
                ax.autoscale_view()
        
        except queue.Empty:
            pass
            
        # Ensure the GUI stays responsive and draws new frames
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    # 4. Clean up and Save
    for p in workers:
        p.join()
        
    print("-" * 60)
    print("\nBenchmark complete. All cores finished successfully.")
    plt.ioff() 
    plt.savefig('divergence_vs_perveance_benchmark_multicore.png', dpi=300)
    print("Plot saved as 'divergence_vs_perveance_benchmark_multicore.png'.")
    plt.show() 

# Mandatory guard block for Windows multiprocessing
if __name__ == "__main__":
    run_parallel_benchmark()