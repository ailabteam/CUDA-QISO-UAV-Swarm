# src/QISO_Core.py

class QISO_Optimizer:
    """
    Triển khai thuật toán Quantum-Inspired Particle Swarm Optimization.
    Toàn bộ dữ liệu hạt được lưu trữ và xử lý trên GPU (CuPy).
    """
    def __init__(self, env, N_particles, max_iter, params, cp):
        self.env = env
        self.cp = cp
        self.N_particles = N_particles
        self.max_iter = max_iter
        self.dim = env.N_uavs * env.N_waypoints * 3 # Kích thước không gian giải pháp
        
        # Tham số QISO
        self.C1 = params.get('C1', 1.5)
        self.C2 = params.get('C2', 1.5)
        self.L_factor = params.get('L_factor', 0.5) # Hệ số dịch chuyển Lượng tử L

        # Khởi tạo dữ liệu trên GPU
        self.positions = self.cp.random.uniform(
            low=env.params['min_bound'], 
            high=env.params['max_bound'], 
            size=(N_particles, self.dim)
        )
        
        # P_best: Vị trí tốt nhất cá nhân
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = self.cp.full(N_particles, self.cp.inf)
        
        # G_best: Vị trí tốt nhất toàn cục (khởi tạo tạm)
        self.gbest_position = None
        self.gbest_fitness = self.cp.inf

    def optimize(self):
        for t in range(self.max_iter):
            # 1. Đánh giá Hàm Mục tiêu (Chạy trên GPU qua Environment)
            current_fitness = self.env.evaluate_fitness(self.positions) # Trả về mảng CPU

            # Chuyển fitness về lại GPU để so sánh
            cp_current_fitness = self.cp.array(current_fitness) 
            
            # 2. Cập nhật P_best
            better_indices = cp_current_fitness < self.pbest_fitness
            self.pbest_fitness[better_indices] = cp_current_fitness[better_indices]
            # Cập nhật vị trí pbest (CuPy indexing)
            self.pbest_positions[better_indices] = self.positions[better_indices]

            # 3. Cập nhật G_best
            min_fitness_index = self.cp.argmin(cp_current_fitness)
            if cp_current_fitness[min_fitness_index] < self.gbest_fitness:
                self.gbest_fitness = cp_current_fitness[min_fitness_index]
                self.gbest_position = self.positions[min_fitness_index].copy()
            
            # 4. Cập nhật Vị trí Lượng tử (Q-Update) - CHÍNH Ở ĐÂY!
            self._quantum_update()
            
            if t % 50 == 0:
                print(f"Iteration {t}/{self.max_iter} - G_Best Fitness: {self.gbest_fitness.item()}")
                
        return self.gbest_position.get(), self.gbest_fitness.item()

    def _quantum_update(self):
        """
        Thực hiện công thức cập nhật vị trí Quantum-Inspired.
        Tất cả phép toán phải là CuPy để chạy trên GPU.
        """
        # Tính toán m_best (Trung tâm Lượng tử)
        # Sử dụng CuPy để tính m_best cho toàn bộ quần thể song song
        r1 = self.cp.random.uniform(0, self.C1, size=(self.N_particles, 1))
        r2 = self.cp.random.uniform(0, self.C2, size=(self.N_particles, 1))
        
        m_best = (r1 * self.pbest_positions + r2 * self.gbest_position) / (r1 + r2)
        
        # Lấy số ngẫu nhiên cho dịch chuyển Lượng tử U(0, 1)
        u = self.cp.random.uniform(0.001, 1.0, size=(self.N_particles, self.dim))
        
        # Tính toán Hệ số Dịch chuyển L (L_factor có thể điều chỉnh theo t)
        L = self.L_factor * self.cp.abs(self.gbest_position - self.positions) # Biến thể QISO phổ biến
        
        # Tính toán Phân bố Lượng tử (dựa trên ln(1/u))
        quantum_jump = 0.5 * L * self.cp.log(1.0 / u)

        # Quyết định hướng nhảy (±) - Sử dụng mask ngẫu nhiên
        flip_mask = self.cp.random.rand(self.N_particles, self.dim) < 0.5
        
        # Cập nhật vị trí mới (CuPy vectorized operation)
        self.positions = m_best + self.cp.where(flip_mask, quantum_jump, -quantum_jump)
        
        # Áp dụng ràng buộc không gian (giới hạn min/max)
        min_b = self.env.params['min_bound']
        max_b = self.env.params['max_bound']
        
        self.positions = self.cp.clip(self.positions, min_b, max_b)
