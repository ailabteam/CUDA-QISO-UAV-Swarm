# src/QISO_Core.py (Phiên bản cuối: Hỗ trợ SPSO, QISO, Chaos, và Log History)

import cupy as cp
import numpy as np

class QISO_Optimizer:
    
    def __init__(self, env, N_particles, max_iter, params, cp):
        self.env = env
        self.cp = cp
        self.N_particles = N_particles
        self.max_iter = max_iter
        self.dim = env.N_uavs * env.N_waypoints * 3
        
        # Tham số thuật toán
        self.C1 = params.get('C1', 1.5)
        self.C2 = params.get('C2', 1.5)
        self.is_qiso = params.get('is_qiso', True)
        
        # Tham số QISO
        self.L_factor = params.get('L_factor', 0.5)

        # Tham số SPSO
        self.W = params.get('W', 0.7)
        
        # Tham số Chaos (cho QISO)
        self.chaotic_map_param = params.get('chaos_mu', 4.0)
        # Khởi tạo trạng thái hỗn loạn (ví dụ: Logistic Map)
        self.chaos_state = cp.array([0.5], dtype=cp.float32)

        # Ràng buộc không gian
        self.min_b = env.params['min_bound']
        self.max_b = env.params['max_bound']

        # Khởi tạo Vị trí ban đầu
        self.positions = self.cp.random.uniform(
            low=self.min_b, 
            high=self.max_b, 
            size=(N_particles, self.dim)
        ).astype(self.cp.float32)
        
        # Khởi tạo Vận tốc nếu là SPSO
        if not self.is_qiso:
            self.velocities = self.cp.zeros((N_particles, self.dim), dtype=self.cp.float32)
        
        # Gán Waypoint 1 (Start Position) cố định (sẽ được thực hiện trong hàm khởi tạo)
        self._apply_start_position_constraint_init()

        # P_best/G_best và Log
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = self.cp.full(N_particles, self.cp.inf)
        self.gbest_position = None
        self.gbest_fitness = self.cp.inf
        self.gbest_history = [] # Log hội tụ

    def _apply_start_position_constraint_init(self):
        # Đảm bảo Waypoint 1 luôn cố định (chỉ chạy một lần lúc khởi tạo)
        for i in range(self.env.N_uavs):
             start_coords = self.cp.array(self.env.params['start_pos'][i], dtype=self.cp.float32)
             self.positions[:, i*self.env.N_waypoints*3 : i*self.env.N_waypoints*3 + 3] = start_coords

    def optimize(self):
        algo_type = "QISO" if self.is_qiso else "SPSO"
        
        for t in range(self.max_iter):
            
            # Tính toán trạng thái Hỗn loạn MỚI (Logistic Map)
            if self.is_qiso:
                # Mu * x * (1 - x). Tính toán trên GPU cho hiệu suất.
                self.chaos_state = self.chaotic_map_param * self.chaos_state * (1.0 - self.chaos_state)

            current_fitness = self.env.evaluate_fitness(self.positions)
            cp_current_fitness = self.cp.array(current_fitness) 
            
            # Cập nhật P_best
            better_indices = cp_current_fitness < self.pbest_fitness
            self.pbest_fitness = self.cp.where(better_indices, cp_current_fitness, self.pbest_fitness)
            self.pbest_positions[better_indices] = self.positions[better_indices]

            # Cập nhật G_best
            min_idx = self.cp.argmin(cp_current_fitness)
            if cp_current_fitness[min_idx] < self.gbest_fitness:
                self.gbest_fitness = cp_current_fitness[min_idx]
                self.gbest_position = self.positions[min_idx].copy()
            
            # GHI LOG HỘI TỤ
            self.gbest_history.append(self.gbest_fitness.item())

            # Cập nhật vị trí (SPSO hoặc QISO)
            self._update()
            
            if t % 50 == 0:
                print(f"[{algo_type}] Iteration {t}/{self.max_iter} - G_Best Fitness: {self.gbest_fitness.item():.2f}")
                
        return self.gbest_position.get(), self.gbest_fitness.item(), self.gbest_history

    def _update(self):
        """ Chọn cơ chế cập nhật (QISO hoặc SPSO). """
        if self.is_qiso:
            self._quantum_update()
        else:
            self._standard_update()

    def _quantum_update(self):
        """ Thực hiện công thức cập nhật vị trí Quantum-Inspired Hỗn loạn. """
        
        r1 = self.cp.random.uniform(0, self.C1, size=(self.N_particles, 1))
        r2 = self.cp.random.uniform(0, self.C2, size=(self.N_particles, 1))
        
        gbest_expanded = self.cp.tile(self.gbest_position, (self.N_particles, 1))
        m_best = (r1 * self.pbest_positions + r2 * gbest_expanded) / (r1 + r2)
        
        # Lấy tham số điều chỉnh L từ trạng thái Hỗn loạn
        dynamic_L_factor = self.chaos_state.item() 
        
        u = self.cp.random.uniform(0.001, 1.0, size=(self.N_particles, self.dim))
        
        # Lượng tử hóa L: L = dynamic_L_factor * |G_best - X|
        L = dynamic_L_factor * self.cp.abs(gbest_expanded - self.positions) 
        quantum_jump = 0.5 * L * self.cp.log(1.0 / u)

        flip_mask = self.cp.random.rand(self.N_particles, self.dim) < 0.5
        self.positions = m_best + self.cp.where(flip_mask, quantum_jump, -quantum_jump)
        
        # Áp dụng ràng buộc không gian và ràng buộc cứng
        self.positions = self.cp.clip(self.positions, self.min_b, self.max_b)
        self._apply_start_position_constraint()

    def _standard_update(self):
        """ Thực hiện công thức cập nhật vị trí và vận tốc SPSO cổ điển. """
        
        r1 = self.cp.random.uniform(0, self.C1, size=(self.N_particles, 1))
        r2 = self.cp.random.uniform(0, self.C2, size=(self.N_particles, 1))
        
        gbest_expanded = self.cp.tile(self.gbest_position, (self.N_particles, 1))

        # Cập nhật Vận tốc
        cognitive = r1 * (self.pbest_positions - self.positions)
        social = r2 * (gbest_expanded - self.positions)
        self.velocities = self.W * self.velocities + cognitive + social
        
        # Cập nhật Vị trí
        self.positions += self.velocities
        
        # Áp dụng ràng buộc không gian và ràng buộc cứng
        self.positions = self.cp.clip(self.positions, self.min_b, self.max_b)
        self._apply_start_position_constraint()
        
    def _apply_start_position_constraint(self):
        # Đảm bảo Waypoint 1 luôn cố định (chạy trong mỗi lần lặp)
        for i in range(self.env.N_uavs):
             start_coords = self.cp.array(self.env.params['start_pos'][i], dtype=self.cp.float32)
             self.positions[:, i*self.env.N_waypoints*3 : i*self.env.N_waypoints*3 + 3] = start_coords
