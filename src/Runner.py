# src/Runner.py

import cupy as cp
import time
import matplotlib.pyplot as plt
import numpy as np

# Sử dụng cú pháp import package
from src.Environment import UAV_Environment
from src.QISO_Core import QISO_Optimizer
from data.config_scenario1 import SCENARIO_CONFIG

def run_simulation(config):
    
    qiso_params = config['qiso_params']
    
    print(f"--- Starting Simulation: {qiso_params['simulation_name']} ---")
    
    # 1. Khởi tạo Môi trường (Truyền toàn bộ config)
    env = UAV_Environment(
        N_uavs=config['sim_params']['N_uavs'], 
        N_waypoints=config['sim_params']['N_waypoints'], 
        config=config, # Truyền toàn bộ SCENARIO_CONFIG
        cp=cp
    )
    
    # 2. Khởi tạo QISO Optimizer
    optimizer = QISO_Optimizer(
        env=env, 
        N_particles=qiso_params['N_particles'], 
        max_iter=qiso_params['max_iter'], 
        params=qiso_params, 
        cp=cp
    )
    
    # 3. Chạy Tối ưu hóa
    start_time = time.time()
    gbest_position, gbest_fitness = optimizer.optimize()
    end_time = time.time()
    
    print("\n--- Optimization Complete ---")
    print(f"Final G_Best Fitness: {gbest_fitness:.4f}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    return gbest_position, gbest_fitness, env.N_uavs, env.N_waypoints, qiso_params

def visualize_results(gbest_pos, N_uavs, N_waypoints, config, qiso_params):
    
    # ... (giữ nguyên logic visualize) ...
    path = gbest_pos.reshape(N_uavs, N_waypoints, 3)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Vẽ Chướng ngại vật
    obs_data = np.array(config['obstacles_data'])
    for i in range(obs_data.shape[0]):
        center = obs_data[i, :3]
        radius = obs_data[i, 3]
        ax.scatter(center[0], center[1], center[2], color='red', marker='o', s=100)

    # Vẽ Quỹ đạo UAV
    for i in range(N_uavs):
        ax.plot(path[i, :, 0], path[i, :, 1], path[i, :, 2], marker='.', 
                linestyle='-', label=f'UAV {i+1}')
        
        # Điểm bắt đầu
        start_pos = config['sim_params']['start_pos'][i]
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], 
                   marker='s', color='green', s=50, label='Start' if i == 0 else "")
        # Điểm kết thúc (Waypoint cuối cùng)
        ax.scatter(path[i, -1, 0], path[i, -1, 1], path[i, -1, 2], 
                   marker='x', color='blue', s=50, label='End' if i == 0 else "")
    
    # Vẽ các mục tiêu nhiệm vụ (Mission Targets)
    mission_targets = np.array(config['mission_targets'])
    for target in mission_targets:
        ax.scatter(target[0], target[1], target[2], marker='*', color='gold', s=150)
    
    # Thiết lập giới hạn
    bounds = config['sim_params']['dimensions']
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_zlim(0, bounds[2])
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"QISO Optimized Paths - {qiso_params['simulation_name']}")
    
    output_filename = f"results/{qiso_params['simulation_name']}_path.pdf"
    plt.savefig(output_filename, format='pdf')
    plt.close(fig) 
    print(f"Saved visualization to {output_filename}")


if __name__ == "__main__":
    
    # Thiết lập Matplotlib để chạy non-GUI
    plt.switch_backend('Agg') 
    
    # Chạy mô phỏng
    gbest_pos, gbest_fitness, N_uavs, N_waypoints, qiso_params = run_simulation(SCENARIO_CONFIG)
    
    # Trực quan hóa kết quả
    visualize_results(gbest_pos, N_uavs, N_waypoints, SCENARIO_CONFIG, qiso_params)
