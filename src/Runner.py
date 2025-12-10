# src/Runner.py

import cupy as cp
import time
import matplotlib.pyplot as plt
import numpy as np

from src.Environment import UAV_Environment
from src.QISO_Core import QISO_Optimizer
from data.config_scenario1 import SCENARIO_CONFIG, QISO_PARAMS

def run_simulation(config, qiso_params):
    
    # 1. Khởi tạo Môi trường và QISO
    print(f"--- Starting Simulation: {qiso_params['simulation_name']} ---")
    
    # Tích hợp QISO_PARAMS vào SIM_PARAMS để truyền vào Environment
    config['sim_params']['QISO_PARAMS'] = qiso_params

    # Khởi tạo Environment
    env = UAV_Environment(
        N_uavs=config['sim_params']['N_uavs'], 
        N_waypoints=config['sim_params']['N_waypoints'], 
        sim_params=config['sim_params'], 
        cp=cp
    )
    
    # Khởi tạo QISO Optimizer
    optimizer = QISO_Optimizer(
        env=env, 
        N_particles=qiso_params['N_particles'], 
        max_iter=qiso_params['max_iter'], 
        params=qiso_params, 
        cp=cp
    )
    
    # 2. Chạy Tối ưu hóa
    start_time = time.time()
    
    gbest_position, gbest_fitness = optimizer.optimize()
    
    end_time = time.time()
    
    print("\n--- Optimization Complete ---")
    print(f"Final G_Best Fitness: {gbest_fitness:.4f}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    return gbest_position, gbest_fitness, env.N_uavs, env.N_waypoints

def visualize_results(gbest_pos, N_uavs, N_waypoints, config):
    """
    Trực quan hóa quỹ đạo 3D (dùng Matplotlib non-GUI backend) và lưu PDF.
    """
    
    # Chuyển quỹ đạo tối ưu về dạng dễ hình dung (N_uavs, N_waypoints, 3)
    path = gbest_pos.reshape(N_uavs, N_waypoints, 3)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Vẽ Chướng ngại vật (Spheres - đơn giản hóa)
    obs_data = np.array(config['obstacles_data'])
    
    for i in range(obs_data.shape[0]):
        center = obs_data[i, :3]
        radius = obs_data[i, 3]
        
        # Vẽ tâm chướng ngại vật
        ax.scatter(center[0], center[1], center[2], color='red', marker='o', s=100)
        # (Vẽ hình cầu phức tạp hơn, tạm thời chỉ vẽ tâm)

    # Vẽ Quỹ đạo UAV
    for i in range(N_uavs):
        ax.plot(path[i, :, 0], path[i, :, 1], path[i, :, 2], marker='o', 
                linestyle='-', label=f'UAV {i+1}')
        
        # Điểm bắt đầu
        ax.scatter(path[i, 0, 0], path[i, 0, 1], path[i, 0, 2], 
                   marker='s', color='green', s=50, label='Start' if i == 0 else "")
        # Điểm kết thúc
        ax.scatter(path[i, -1, 0], path[i, -1, 1], path[i, -1, 2], 
                   marker='x', color='blue', s=50, label='End' if i == 0 else "")
    
    # Thiết lập giới hạn
    bounds = config['sim_params']['dimensions']
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_zlim(0, bounds[2])
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"QISO Optimized Paths - {qiso_params['simulation_name']}")
    
    # Lưu kết quả
    output_filename = f"results/{qiso_params['simulation_name']}_path.pdf"
    plt.savefig(output_filename, format='pdf')
    print(f"Saved visualization to {output_filename}")
    # Đóng figure để tránh rò rỉ bộ nhớ
    plt.close(fig) 


if __name__ == "__main__":
    
    # Thiết lập Matplotlib để chạy non-GUI
    plt.switch_backend('Agg') 
    
    # Chạy mô phỏng
    gbest_pos, gbest_fitness, N_uavs, N_waypoints = run_simulation(SCENARIO_CONFIG, QISO_PARAMS)
    
    # Trực quan hóa kết quả
    visualize_results(gbest_pos, N_uavs, N_waypoints, SCENARIO_CONFIG)
