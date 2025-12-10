# src/Runner.py (Phiên bản cuối: Hỗ trợ Log, Legend, và Convergence Plot)

import cupy as cp
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from src.Environment import UAV_Environment
from src.QISO_Core import QISO_Optimizer
from data.config_scenario1 import SCENARIO_CONFIG as CONFIG_1
# Sẽ import CONFIG_2 trong phần main

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
    # Nhận thêm history
    gbest_position, gbest_fitness, history = optimizer.optimize() 
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
        "convergence_history": history
    }
    
    return gbest_position, metrics, env.N_uavs, env.N_waypoints, config

def save_metrics(metrics):
    filename = f"results/{metrics['scenario']}_{metrics['algorithm']}_metrics.json"
    
    # Chuyển history thành list Python chuẩn trước khi lưu JSON
    metrics_to_save = metrics.copy()
    metrics_to_save['convergence_history'] = [float(f) for f in metrics_to_save['convergence_history']]
    
    with open(filename, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"Saved metrics to {filename}")

def visualize_results(gbest_pos, N_uavs, N_waypoints, config, metrics):
    
    path = gbest_pos.reshape(N_uavs, N_waypoints, 3)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Vẽ Chướng ngại vật và Mission Targets (Đảm bảo Legend)
    obs_data = np.array(config['obstacles_data'])
    
    # 1. Chướng ngại vật Tĩnh (Màu Đỏ, marker tròn)
    is_first_obs = True
    for i in range(obs_data.shape[0]):
        center = obs_data[i, :3]
        ax.scatter(center[0], center[1], center[2], color='red', marker='o', s=100, 
                   label='Static Obstacles' if is_first_obs else None)
        is_first_obs = False

    # 2. Mục tiêu Nhiệm vụ (Màu Vàng, marker Ngôi sao)
    mission_targets = np.array(config['mission_targets'])
    is_first_target = True
    for target in mission_targets:
        ax.scatter(target[0], target[1], target[2], marker='*', color='gold', s=200,
                   label='Mission Target' if is_first_target else None)
        is_first_target = False
    
    # 3. Quỹ đạo UAV
    for i in range(N_uavs):
        # Đường đi
        ax.plot(path[i, :, 0], path[i, :, 1], path[i, :, 2], 
                linestyle='-', linewidth=1.5, label=f'UAV {i+1}' if i == 0 else None)
        
        # Điểm bắt đầu (marker vuông Xanh lá)
        start_pos = config['sim_params']['start_pos'][i]
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], 
                   marker='s', color='green', s=70, label='Start Position' if i == 0 else None)
        
        # Điểm kết thúc (marker X Xanh dương)
        ax.scatter(path[i, -1, 0], path[i, -1, 1], path[i, -1, 2], 
                   marker='x', color='blue', s=70, label='End Waypoint' if i == 0 else None)
    
    # Thiết lập giới hạn
    bounds = config['sim_params']['dimensions']
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_zlim(0, bounds[2])
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"{metrics['algorithm']} Optimized Paths - {metrics['scenario']}")
    
    # Thêm Legend
    ax.legend(loc='upper right', fontsize='small')
    
    output_filename = f"results/{metrics['scenario']}_{metrics['algorithm']}_path.pdf"
    plt.savefig(output_filename, format='pdf')
    plt.close(fig) 
    print(f"Saved visualization to {output_filename}")

def visualize_convergence(history_spso, history_qiso, scenario_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Chuyển đổi list (nếu cần) và vẽ
    ax.plot(history_spso, label='SPSO (Baseline)', color='blue', linewidth=2)
    ax.plot(history_qiso, label='QISO-Chaos (Proposed)', color='red', linewidth=2, linestyle='--')
    
    ax.set_title(f'Convergence Analysis - {scenario_name}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('G_Best Fitness')
    ax.legend()
    ax.grid(True)
    
    output_filename = f"results/{scenario_name}_convergence.pdf"
    plt.savefig(output_filename, format='pdf')
    plt.close(fig)
    print(f"Saved convergence plot to {output_filename}")


if __name__ == "__main__":
    
    plt.switch_backend('Agg') 
    
    # CHẠY TRÊN KỊCH BẢN 2 (Môi trường động, quy mô lớn hơn)
    from data.config_scenario2 import SCENARIO_CONFIG_2 as CONFIG_TEST
    
    metrics_log = {'spso': None, 'qiso': None}
    history_log = {'spso': None, 'qiso': None}

    # 1. BASELINE: SPSO trên KỊCH BẢN 2
    config_spso = CONFIG_TEST.copy()
    config_spso['qiso_params']['is_qiso'] = False
    config_spso['qiso_params']['simulation_name'] = "Scenario_2_Baseline"
    
    gbest_pos_spso, metrics_spso, N_uavs, N_waypoints, config_used = run_simulation(config_spso)
    save_metrics(metrics_spso)
    visualize_results(gbest_pos_spso, N_uavs, N_waypoints, config_used, metrics_spso)
    metrics_log['spso'] = metrics_spso
    history_log['spso'] = metrics_spso['convergence_history']
    
    print("-" * 50)
    
    # 2. PROPOSED: QISO (Chaos-Enhanced) trên KỊCH BẢN 2
    config_qiso = CONFIG_TEST.copy()
    config_qiso['qiso_params']['is_qiso'] = True
    config_qiso['qiso_params']['simulation_name'] = "Scenario_2_QISO_Chaos"
    config_qiso['qiso_params']['chaos_mu'] = 4.0
    
    gbest_pos_qiso, metrics_qiso, N_uavs, N_waypoints, config_used = run_simulation(config_qiso)
    save_metrics(metrics_qiso)
    visualize_results(gbest_pos_qiso, N_uavs, N_waypoints, config_used, metrics_qiso)
    metrics_log['qiso'] = metrics_qiso
    history_log['qiso'] = metrics_qiso['convergence_history']

    # 3. TRỰC QUAN HÓA HỘI TỤ
    visualize_convergence(history_log['spso'], history_log['qiso'], "Scenario_2_Comparison")
