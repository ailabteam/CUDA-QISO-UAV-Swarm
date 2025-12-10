# src/Runner.py

import cupy as cp
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from src.Environment import UAV_Environment
from src.QISO_Core import QISO_Optimizer
from data.config_scenario1 import SCENARIO_CONFIG as CONFIG_1
# Tạm thời import config 1 để chạy thử, sau này sẽ thay đổi bằng tham số đầu vào

def run_simulation(config):

    qiso_params = config['qiso_params']
    algo_type = "QISO" if qiso_params.get('is_qiso', True) else "SPSO"
    qiso_params['algo_type'] = algo_type

    print(f"--- Starting Simulation: {qiso_params['simulation_name']} [{algo_type}] ---")

    env = UAV_Environment(
        N_uavs=config['sim_params']['N_uavs'],
        N_waypoints=config['sim_params']['N_waypoints'],
        config=config,
        cp=cp
    )

    optimizer = QISO_Optimizer(
        env=env,
        N_particles=qiso_params['N_particles'],
        max_iter=qiso_params['max_iter'],
        params=qiso_params,
        cp=cp
    )

    start_time = time.time()
    gbest_position, gbest_fitness = optimizer.optimize()
    end_time = time.time()

    total_time = end_time - start_time

    print("\n--- Optimization Complete ---")
    print(f"Final G_Best Fitness: {gbest_fitness:.4f}")
    print(f"Total time: {total_time:.2f} seconds")

    metrics = {
        "algorithm": algo_type,
        "scenario": qiso_params['simulation_name'],
        "gbest_fitness": float(gbest_fitness),
        "total_time_s": total_time,
        "n_particles": qiso_params['N_particles'],
        "max_iter": qiso_params['max_iter'],
        "N_uavs": config['sim_params']['N_uavs'],
        # Thêm các metrics khác sau này (ví dụ: log hội tụ)
    }

    return gbest_position, metrics, env.N_uavs, env.N_waypoints, config

def save_metrics(metrics):
    filename = f"results/{metrics['scenario']}_{metrics['algorithm']}_metrics.json"
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {filename}")

def visualize_results(gbest_pos, N_uavs, N_waypoints, config, metrics):

    path = gbest_pos.reshape(N_uavs, N_waypoints, 3)

    # ... (Logic vẽ đồ thị 3D giữ nguyên) ...
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Vẽ Chướng ngại vật
    obs_data = np.array(config['obstacles_data'])
    # Sử dụng cờ để đảm bảo chỉ vẽ 1 legend entry cho chướng ngại vật và nhiệm vụ
    is_first_obs = True
    for i in range(obs_data.shape[0]):
        center = obs_data[i, :3]
        ax.scatter(center[0], center[1], center[2], color='red', marker='o', s=100, 
                   label='Static Obstacles' if is_first_obs else "")
        is_first_obs = False

    # Vẽ Quỹ đạo UAV
    for i in range(N_uavs):
        # Marker='.' cho các điểm Waypoint
        ax.plot(path[i, :, 0], path[i, :, 1], path[i, :, 2], marker='.', 
                linestyle='-', label=f'UAV {i+1}' if i < 1 else "") # Chỉ chú thích 1 UAV
        
        # Điểm bắt đầu
        start_pos = config['sim_params']['start_pos'][i]
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], 
                   marker='s', color='green', s=50, label='Start Position' if i == 0 else "")


    # Vẽ các mục tiêu nhiệm vụ (Mission Targets)
    mission_targets = np.array(config['mission_targets'])
    is_first_target = True
    for target in mission_targets:
        ax.scatter(target[0], target[1], target[2], marker='*', color='gold', s=150,
                   label='Mission Target' if is_first_target else "") # Ngôi sao vàng
        is_first_target = False

    bounds = config['sim_params']['dimensions']
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_zlim(0, bounds[2])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"{metrics['algorithm']} Optimized Paths - {metrics['scenario']}")

    # THÊM LEGEND
    ax.legend(loc='best', fontsize='small')


    output_filename = f"results/{metrics['scenario']}_{metrics['algorithm']}_path.pdf"
    plt.savefig(output_filename, format='pdf')
    plt.close(fig)
    print(f"Saved visualization to {output_filename}")



if __name__ == "__main__":
    
    # Thiết lập Matplotlib để chạy non-GUI
    plt.switch_backend('Agg') 
    
    # Nhập cấu hình cho KỊCH BẢN 2
    from data.config_scenario2 import SCENARIO_CONFIG_2 as CONFIG_TEST
    
    # 1. BASELINE: SPSO trên KỊCH BẢN 2
    config_spso = CONFIG_TEST.copy()
    config_spso['qiso_params']['is_qiso'] = False
    config_spso['qiso_params']['simulation_name'] = "Scenario_2_Baseline"
    
    gbest_pos_spso, metrics_spso, N_uavs, N_waypoints, config_used = run_simulation(config_spso)
    save_metrics(metrics_spso)
    visualize_results(gbest_pos_spso, N_uavs, N_waypoints, config_used, metrics_spso)
    
    print("-" * 50)
    
    # 2. PROPOSED: QISO trên KỊCH BẢN 2
    config_qiso = CONFIG_TEST.copy()
    config_qiso['qiso_params']['is_qiso'] = True
    config_qiso['qiso_params']['simulation_name'] = "Scenario_2_QISO"

    gbest_pos_qiso, metrics_qiso, N_uavs, N_waypoints, config_used = run_simulation(config_qiso)
    save_metrics(metrics_qiso)
    visualize_results(gbest_pos_qiso, N_uavs, N_waypoints, config_used, metrics_qiso)
