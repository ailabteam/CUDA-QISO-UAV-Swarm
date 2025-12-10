# src/Environment.py (Cập nhật)

import cupy as cp
from src.QISO_CUDA_Kernels import static_collision_kernel, calculate_f1_cupy, calculate_f3_cupy, THREADS_PER_BLOCK

class UAV_Environment:
    
    def __init__(self, N_uavs, N_waypoints, sim_params, cp):
        self.cp = cp
        self.N_uavs = N_uavs
        self.N_waypoints = N_waypoints
        self.params = sim_params
        
        # Đặt các tham số môi trường lên GPU
        self.obstacles = self.cp.array(sim_params['obstacles_data'], dtype=self.cp.float32) 
        self.mission_targets = self.cp.array(sim_params['mission_targets'], dtype=self.cp.float32)
        
        # Tính toán cấu hình CUDA grid
        N_particles = sim_params['QISO_PARAMS']['N_particles']
        self.blocks = (N_particles + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        
    def evaluate_fitness(self, positions):
        N_particles = positions.shape[0]
        fitness = self.cp.zeros(N_particles, dtype=self.cp.float32)

        # Tái cấu trúc mảng vị trí cho tính toán f1
        reshaped_pos = positions.reshape(N_particles, self.N_uavs, self.N_waypoints, 3)
        
        # --- Tính toán F1 (Quãng đường/Năng lượng) - CuPy Vectorized ---
        distance_cost = calculate_f1_cupy(reshaped_pos)
        
        # --- Tính toán F2 (Va chạm Tĩnh) - Numba CUDA Kernel ---
        collision_penalty_f2 = self.cp.zeros(N_particles, dtype=self.cp.float32)
        
        # Gọi Kernel Numba CUDA
        static_collision_kernel[self.blocks, THREADS_PER_BLOCK](
            positions, 
            self.obstacles, 
            self.N_uavs, 
            self.N_waypoints, 
            self.obstacles.shape[0], # Số lượng chướng ngại vật
            5000.0,                   # Giá trị phạt cứng
            collision_penalty_f2
        )
        
        # --- Tính toán F3 (Nhiệm vụ) - CuPy Vectorized ---
        task_penalty = calculate_f3_cupy(reshaped_pos, self.mission_targets)
        
        # Áp dụng trọng số
        w1, w2, w3 = self.params['weights']
        fitness = w1 * distance_cost + w2 * collision_penalty_f2 + w3 * task_penalty
        
        return fitness.get() # Trả về mảng NumPy trên CPU
