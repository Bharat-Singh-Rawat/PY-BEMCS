import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import queue
from physics_engine import DigitalTwinSimulator

def worker_sweep(gap, Va_sweep, result_queue):
    for Va in Va_sweep:
        sim = DigitalTwinSimulator()
        params = {
            'sim_mode': 'Both',
            'ts': 1, 'ta': 1, 'rs': 1.0, 'ra': 0.6,
            'cham_s': 0, 'cham_a': 0,
            'Vs': 1650, 'Va': Va, # Sweeping Va
            'gap': gap,
            'n0_plasma': 1e17,    # Fixed optimal density
            'Te_up': 3.0, 'Ti': 0.1,
            'neut_rate': 0, 'n0': 0 
        }
        
        sim.build_domain(params)
        steady_state_steps = 500
        pot_history = []
        
        for step_idx in range(1, steady_state_steps + 1):
            _, min_pot, _, _, _ = sim.step(params)
            
            pot_history.append(min_pot)
            if len(pot_history) > 50:
                pot_history.pop(0) 
            
            if step_idx % 50 == 0 or step_idx == 1:
                print(f"[Core Gap {gap}mm | Va: {Va:.0f}V] Step {step_idx}/{steady_state_steps} -> Saddle: {min_pot:.2f}V", flush=True)

            if step_idx >= 250 and len(pot_history) == 50:
                if np.std(pot_history) < 0.1: 
                    break

        final_pot = np.mean(pot_history) if len(pot_history) > 0 else min_pot
        result_queue.put(('data', gap, Va, final_pot))
        
    result_queue.put(('done', gap))

def run_ebs_benchmark():
    grid_gaps = [0.5, 0.75, 1.0] 
    Va_sweep = np.linspace(-400, 0, 15) 
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('EBS Limit: Centerline Saddle Point vs. Accel Voltage')
    ax.set_xlabel('Accelerator Grid Voltage (V)')
    ax.set_ylabel('Saddle Point Potential (V)')
    ax.grid(True)
    ax.axhline(-5, color='red', linestyle='--', alpha=0.5, label='EBS Risk Threshold (~ -5V)')
    
    lines_sim = {}
    data_store = {gap: {'Va': [], 'pot': []} for gap in grid_gaps}
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for i, gap in enumerate(grid_gaps):
        lines_sim[gap], = ax.plot([], [], marker='o', color=colors[i], label=f'Gap = {gap} mm')
        
    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events() 
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    workers = []
    
    for gap in grid_gaps:
        p = mp.Process(target=worker_sweep, args=(gap, Va_sweep, result_queue))
        p.start()
        workers.append(p)
        
    completed_workers = 0
    while completed_workers < len(grid_gaps):
        try:
            msg = result_queue.get(timeout=0.1)
            if msg[0] == 'done':
                completed_workers += 1
            elif msg[0] == 'data':
                _, gap, Va, pot = msg
                data_store[gap]['Va'].append(Va)
                data_store[gap]['pot'].append(pot)
                
                sort_idx = np.argsort(data_store[gap]['Va'])
                lines_sim[gap].set_data(np.array(data_store[gap]['Va'])[sort_idx], 
                                        np.array(data_store[gap]['pot'])[sort_idx])
                ax.relim()
                ax.autoscale_view()
        except queue.Empty:
            pass
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    for p in workers: p.join()
    plt.ioff() 
    plt.savefig('ebs_benchmark_multicore.png', dpi=300)
    plt.show() 

if __name__ == "__main__":
    run_ebs_benchmark()