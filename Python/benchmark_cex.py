import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import queue
from physics_engine import DigitalTwinSimulator

def worker_sweep(gap, nn_sweep, result_queue):
    for nn in nn_sweep:
        sim = DigitalTwinSimulator()
        params = {
            'sim_mode': 'Both',
            'ts': 1, 'ta': 1, 'rs': 1.0, 'ra': 0.6,
            'cham_s': 0, 'cham_a': 0,
            'Vs': 1650, 'Va': -350, 
            'gap': gap,
            'n0_plasma': 1e17,   # Fixed optimal perveance
            'Te_up': 3.0, 'Ti': 0.1,
            'neut_rate': 0, 
            'n0': nn,            # Sweeping Neutral Density
            'Tn': 300,           # Neutral gas temperature (K)
            'Accel': 1.0,        # Erosion multiplier
            'Thresh': 1e9        # Keep high so grid doesn't break during benchmark
        }
        
        # NEW: Construct the expected grids array for the physics engine
        params['grids'] = [
            {'V': params['Vs'], 't': params['ts'], 'gap': params['gap'], 'r': params['rs'], 'cham': params['cham_s']},
            {'V': params['Va'], 't': params['ta'], 'gap': 1.0, 'r': params['ra'], 'cham': params['cham_a']}
        ]
        
        sim.build_domain(params)
        steady_state_steps = 500
        
        for step_idx in range(1, steady_state_steps + 1):
            sim.step(params)
            
            if step_idx % 50 == 0 or step_idx == 1:
                # Calculate normalized damage rate
                current_damage = np.sum(sim.damage_map[sim.mask_grids[1]]) / step_idx
                print(f"[Core Gap {gap}mm | nn: {nn:.1e}] Step {step_idx}/{steady_state_steps} -> Erosion Rate: {current_damage:.2e}", flush=True)

        # Final Erosion Rate (Damage per timestep)
        final_erosion_rate = np.sum(sim.damage_map[sim.mask_grids[1]]) / steady_state_steps
        result_queue.put(('data', gap, nn, final_erosion_rate))
        
    result_queue.put(('done', gap))

def run_cex_benchmark():
    grid_gaps = [0.5, 0.75, 1.0] 
    nn_sweep = np.linspace(1e18, 5e19, 10) 
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('CEX Collisions: Accel Grid Erosion vs. Neutral Gas Density')
    ax.set_xlabel('Background Neutral Gas Density (m^-3)')
    ax.set_ylabel('Normalized Accel Grid Erosion Rate (Damage / dt)')
    ax.grid(True)
    
    lines_sim = {}
    data_store = {gap: {'nn': [], 'erosion': []} for gap in grid_gaps}
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
        p = mp.Process(target=worker_sweep, args=(gap, nn_sweep, result_queue))
        p.start()
        workers.append(p)
        
    completed_workers = 0
    while completed_workers < len(grid_gaps):
        try:
            msg = result_queue.get(timeout=0.1)
            if msg[0] == 'done':
                completed_workers += 1
            elif msg[0] == 'data':
                _, gap, nn, erosion = msg
                data_store[gap]['nn'].append(nn)
                data_store[gap]['erosion'].append(erosion)
                
                sort_idx = np.argsort(data_store[gap]['nn'])
                lines_sim[gap].set_data(np.array(data_store[gap]['nn'])[sort_idx], 
                                        np.array(data_store[gap]['erosion'])[sort_idx])
                ax.relim()
                ax.autoscale_view()
        except queue.Empty:
            pass
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    for p in workers: p.join()
    plt.ioff() 
    plt.savefig('cex_benchmark_multicore.png', dpi=300)
    plt.show() 

if __name__ == "__main__":
    run_cex_benchmark()