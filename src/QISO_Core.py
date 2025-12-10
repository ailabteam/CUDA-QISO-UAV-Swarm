# src/QISO_Core.py

import cupy as cp
import numpy as np # Vẫn cần Numpy để khởi tạo từ list/array
# Import Environment để lấy các ràng buộc

class QISO_Optimizer:
    
    def __init__(self, env, N_particles, max_iter, params, cp):
        self.env = env
        self.cp = cp
        self.N_particles = N_particles
        self.max_iter = max_iter
        self.dim = env.N_uavs * env.N_waypoints * 3
        
        self.C1 = params.get('C1', 1.5)
        self.C2 = params.get('C2', 1.5)
        self.L_factor = params.get('L_factor', 0.5)

        # Ràng buộc không gian
        self.min_b = env.params['min_bound']
        self.max_b = env.params['max_bound']

        # Khởi tạo Vị trí: Quan trọng là phải khởi tạo hợp lý.
        # Khởi tạo bằng cách nhân bản và thêm nhiễu xung quanh vị trí START
        start_pos_cpu = np.array(env.params['start_pos']).flatten()
        
        # Tạo mảng vị trí khởi đầu cơ sở (N_particles, total_dim)
        base_positions = np.tile(start_pos_cpu, (N_particles, env.N_waypoints))
        
        # Thêm nhiễu nhỏ (exploration)
        initial_noise = np.random.uniform(-50, 50, size=(N_particles, self.dim))
        
        # Chỉ áp dụng base_position cho waypoint đầu tiên (M=1), 
        # phần còn lại là random trong vùng không gian.
        # Đơn giản hóa: Khởi tạo toàn bộ ngẫu nhiên nhưng trong phạm vi.
        self.positions = self.cp.random.uniform(
            low=self.min_b, 
            high=self.max_b, 
            size=(N_particles, self.dim)
        ).astype(self.cp.float32)
        
        # Khắc phục: Đặt các waypoint đầu tiên là vị trí START_POSITIONS
        for i in range(env.N_uavs):
             # Lấy tọa độ (x, y, z)
            start_coords = cp.array(env.params['start_pos'][i], dtype=self.cp.float32)
            # Gán cho 3 cột đầu tiên của UAV i (Waypoint 1)
            self.positions[:, i*env.N_waypoints*3 : i*env.N_waypoints*3 + 3] = start_coords
        

        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = self.cp.full(N_particles, self.cp.inf)
        
        self.gbest_position = None
        self.gbest_fitness = self.cp.inf

    def optimize(self):
        # ... (giữ nguyên phần optimize) ...
        for t in range(self.max_iter):
            # ... (Phần cập nhật P_best/G_best giữ nguyên) ...
            
            # 1. Đánh giá Hàm Mục tiêu
            current_fitness = self.env.evaluate_fitness(self.positions)
            cp_current_fitness = self.cp.array(current_fitness) 
            
            # 2. Cập nhật P_best
            better_indices = cp_current_fitness < self.pbest_fitness
            self.pbest_fitness = self.cp.where(better_indices, cp_current_fitness, self.pbest_fitness)
            self.pbest_positions[better_indices] = self.positions[better_indices]

            # 3. Cập nhật G_best
            min_idx = self.cp.argmin(cp_current_fitness)
            if cp_current_fitness[min_idx] < self.gbest_fitness:
                self.gbest_fitness = cp_current_fitness[min_idx]
                self.gbest_position = self.positions[min_idx].copy()
            
            # 4. Cập nhật Vị trí Lượng tử (Q-Update)
            self._quantum_update()
            
            if t % 50 == 0:
                print(f"Iteration {t}/{self.max_iter} - G_Best Fitness: {self.gbest_fitness.item():.2f}")
                
        return self.gbest_position.get(), self.gbest_fitness.item()

    def _quantum_update(self):
        """ Thực hiện công thức cập nhật vị trí Quantum-Inspired. """
        
        # ... (Giữ nguyên logic Q-Update) ...
        r1 = self.cp.random.uniform(0, self.C1, size=(self.N_particles, 1))
        r2 = self.cp.random.uniform(0, self.C2, size=(self.N_particles, 1))
        
        # Mở rộng G_best để khớp với kích thước của r1, r2 (N_particles, Dim)
        gbest_expanded = self.cp.tile(self.gbest_position, (self.N_particles, 1))
        
        m_best = (r1 * self.pbest_positions + r2 * gbest_expanded) / (r1 + r2)
        
        u = self.cp.random.uniform(0.001, 1.0, size=(self.N_particles, self.dim))
        
        L = self.L_factor * self.cp.abs(gbest_expanded - self.positions)
        
        quantum_jump = 0.5 * L * self.cp.log(1.0 / u)

        flip_mask = self.cp.random.rand(self.N_particles, self.dim) < 0.5
        
        self.positions = m_best + self.cp.where(flip_mask, quantum_jump, -quantum_jump)
        
        # Áp dụng ràng buộc không gian (giới hạn min/max)
        self.positions = self.cp.clip(self.positions, self.min_b, self.max_b)
        
        # Ràng buộc cứng: Đảm bảo Waypoint 1 (Start Position) luôn cố định
        for i in range(self.env.N_uavs):
             start_coords = cp.array(self.env.params['start_pos'][i], dtype=self.cp.float32)
             self.positions[:, i*self.env.N_waypoints*3 : i*self.env.N_waypoints*3 + 3] = start_coords
