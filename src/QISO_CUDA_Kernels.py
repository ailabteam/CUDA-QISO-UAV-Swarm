# src/QISO_CUDA_Kernels.py

import cupy as cp
from numba import cuda
import numpy as np
import math

# --- Tham số Cấu hình cho CUDA Thread Block ---
THREADS_PER_BLOCK = 256

# Kernel Numba CUDA cho tính toán Va chạm Tĩnh (f2: Penalty Static)
# Vì đây là tính toán trên từng waypoint, Numba Kernel sẽ hiệu quả hơn CuPy nguyên thủy
@cuda.jit
def static_collision_kernel(pos, obstacles, N_uavs, N_waypoints, obstacle_dim, penalty_value, result_f2):
    """
    Tính toán chi phí va chạm tĩnh.
    pos: Mảng vị trí (N_particles, total_dim)
    obstacles: Mảng chướng ngại vật (N_obs, 4: x, y, z, radius)
    result_f2: Mảng đầu ra (N_particles)
    """
    
    # Xác định chỉ mục hạt (particle index) đang được xử lý
    idx = cuda.grid(1) 
    if idx < pos.shape[0]:
        total_penalty = 0.0
        
        # Lặp qua tất cả các waypoint của tất cả các UAV thuộc hạt idx
        for i in range(N_uavs * N_waypoints):
            # Lấy tọa độ waypoint (x, y, z)
            x_w = pos[idx, i * 3 + 0]
            y_w = pos[idx, i * 3 + 1]
            z_w = pos[idx, i * 3 + 2]
            
            # Lặp qua tất cả chướng ngại vật
            for j in range(obstacle_dim):
                # Lấy tọa độ trung tâm và bán kính chướng ngại vật
                x_o = obstacles[j, 0]
                y_o = obstacles[j, 1]
                z_o = obstacles[j, 2]
                r_o = obstacles[j, 3]

                # Tính khoảng cách Euclid giữa waypoint và tâm chướng ngại vật
                dist_sq = (x_w - x_o)**2 + (y_w - y_o)**2 + (z_w - z_o)**2
                
                # Nếu khoảng cách nhỏ hơn bán kính chướng ngại vật
                if dist_sq < r_o**2:
                    # Áp dụng phạt (ví dụ: bình phương khoảng cách vi phạm)
                    # Một hàm phạt đơn giản: penalty_value * (r_o - sqrt(dist_sq))
                    total_penalty += penalty_value * (r_o - math.sqrt(dist_sq)) 
        
        result_f2[idx] = total_penalty


# Hàm tính toán Khoảng cách (f1: Energy/Time Cost) - Tận dụng CuPy Vectorization
def calculate_f1_cupy(reshaped_pos):
    """
    reshaped_pos: (N_particles, N_uavs, N_waypoints, 3)
    """
    # Tính khoảng cách giữa các điểm waypoint liên tiếp
    # (N_particles, N_uavs, N_waypoints - 1, 3)
    diff = reshaped_pos[:, :, 1:, :] - reshaped_pos[:, :, :-1, :]
    
    # Tính bình phương khoảng cách và căn bậc hai
    dist_matrix = cp.sum(diff**2, axis=3)
    
    # Tính tổng quãng đường bay của tất cả UAV trong hạt
    distance_cost = cp.sum(cp.sqrt(dist_matrix), axis=(1, 2))
    return distance_cost

# Placeholder cho Va chạm Động và Nhiệm vụ (sẽ nâng cấp cho Kịch bản 2 & 3)
def calculate_f3_cupy(reshaped_pos, mission_targets):
    """
    Tính toán chi phí nếu không đi qua các mục tiêu nhiệm vụ.
    """
    N_particles = reshaped_pos.shape[0]
    # Hiện tại: Giả định 0 penalty nếu chỉ chạy Kịch bản 1 đơn giản
    # Trong kịch bản phức tạp: cần kiểm tra xem quỹ đạo có đi qua vùng bán kính nhiệm vụ không
    return cp.zeros(N_particles)
