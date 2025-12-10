# src/Environment.py

import cupy as cp
from src.QISO_CUDA_Kernels import static_collision_kernel, calculate_f1_cupy, calculate_f3_cupy, THREADS_PER_BLOCK

class UAV_Environment:
    """
    Định nghĩa môi trường 3D và hàm mục tiêu (Fitness Function).
    Tất cả tính toán phức tạp chạy trên CuPy/Numba.
    """
    def __init__(self, N_uavs, N_waypoints, config, cp):
        self.cp = cp
        self.N_uavs = N_uavs
        self.N_waypoints = N_waypoints
        
        # 'config' là toàn bộ SCENARIO_CONFIG
        self.config = config
        self.params = config['sim_params']
        
        # Khởi tạo dữ liệu trên GPU (Sử dụng key ở tầng trên)
        self.obstacles = self.cp.array(config['obstacles_data'], dtype=self.cp.float32) 
        self.mission_targets = self.cp.array(config['mission_targets'], dtype=self.cp.float32)
        
        # Tính toán cấu hình CUDA grid
        N_particles = config['qiso_params']['N_particles']
        self.blocks = (N_particles + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        
    def evaluate_fitness(self, positions):
        """
        Tính toán hàm mục tiêu F cho toàn bộ quần thể hạt.
        Input: positions (cp.ndarray)
        Output: fitness_values (np.ndarray - CPU)
        """
        N_particles = positions.shape[0]
        
        # Tái cấu trúc mảng vị trí cho tính toán f1
        reshaped_pos = positions.reshape(N_particles, self.N_uavs, self.N_waypoints, 3)
        
        # --- Tính toán F1 (Quãng đường/Năng lượng) ---
        distance_cost = calculate_f1_cupy(reshaped_pos)
        
        # --- Tính toán F2 (Va chạm Tĩnh) ---
        collision_penalty_f2 = self.cp.zeros(N_particles, dtype=self.cp.float32)
        
        # Gọi Kernel Numba CUDA
        # CHÚ Ý: Đảm bảo số lượng chướng ngại vật > 0
        obstacle_count = self.obstacles.shape[0]
        if obstacle_count > 0:
            static_collision_kernel[self.blocks, THREADS_PER_BLOCK](
                positions, 
                self.obstacles, 
                self.N_uavs, 
                self.N_waypoints, 
                obstacle_count,
                self.params['weights'][1] * 100, # Penalty base value (cao để va chạm bị phạt nặng)
                collision_penalty_f2
            )
        
        # --- Tính toán F3 (Nhiệm vụ) ---
        task_penalty = calculate_f3_cupy(reshaped_pos, self.mission_targets)
        
        # Áp dụng trọng số
        w1, w2, w3 = self.params['weights']
        # w2 đã được nhân vào trong kernel, nhưng để đồng nhất, ta dùng trọng số cơ bản ở đây
        fitness = w1 * distance_cost + collision_penalty_f2 + w3 * task_penalty
        
        return fitness.get() # Trả về mảng NumPy trên CPU
