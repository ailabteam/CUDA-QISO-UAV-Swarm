# src/QISO_CUDA_Kernels.py

import cupy as cp
from numba import cuda
import numpy as np
import math

THREADS_PER_BLOCK = 256

@cuda.jit
def static_collision_kernel(pos, obstacles, N_uavs, N_waypoints, obstacle_dim, penalty_value, result_f2):
    
    idx = cuda.grid(1) 
    if idx < pos.shape[0]:
        total_penalty = 0.0
        
        # total_dim = N_uavs * N_waypoints * 3
        
        # Lặp qua tất cả các waypoint của tất cả các UAV thuộc hạt idx
        # Kích thước của pos[idx] là total_dim
        for i in range(N_uavs * N_waypoints):
            # Lấy tọa độ waypoint (x, y, z)
            x_w = pos[idx, i * 3 + 0]
            y_w = pos[idx, i * 3 + 1]
            z_w = pos[idx, i * 3 + 2]
            
            # Lặp qua tất cả chướng ngại vật
            for j in range(obstacle_dim):
                x_o = obstacles[j, 0]
                y_o = obstacles[j, 1]
                z_o = obstacles[j, 2]
                r_o = obstacles[j, 3] # Bán kính an toàn
                
                dist_sq = (x_w - x_o)**2 + (y_w - y_o)**2 + (z_w - z_o)**2
                dist = math.sqrt(dist_sq)

                # Nếu khoảng cách nhỏ hơn bán kính an toàn
                if dist < r_o:
                    # Hàm phạt: Tăng lên khi vi phạm sâu hơn
                    total_penalty += penalty_value * (r_o - dist)
        
        result_f2[idx] = total_penalty


# Hàm tính toán Khoảng cách (f1: Energy/Time Cost) - CuPy Vectorization
def calculate_f1_cupy(reshaped_pos):
    """ reshaped_pos: (N_particles, N_uavs, N_waypoints, 3) """
    
    # Tính hiệu số giữa các waypoint liên tiếp (bỏ qua waypoint cuối)
    diff = reshaped_pos[:, :, 1:, :] - reshaped_pos[:, :, :-1, :]
    
    # Tính bình phương khoảng cách và căn bậc hai
    dist_sq = cp.sum(diff**2, axis=3)
    
    # Tính tổng quãng đường bay của tất cả UAV trong hạt
    distance_cost = cp.sum(cp.sqrt(dist_sq), axis=(1, 2))
    return distance_cost

# Hàm tính toán F3 (Nhiệm vụ) - Hiện tại là placeholder
def calculate_f3_cupy(reshaped_pos, mission_targets):
    N_particles = reshaped_pos.shape[0]
    # Placeholder: Giả định chi phí cố định cho Kịch bản 1 đơn giản
    # Sẽ được phát triển trong Kịch bản 3
    return cp.zeros(N_particles, dtype=cp.float32)
