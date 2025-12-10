# data/config_scenario1.py

# --- Tham số Môi trường Tổng quan ---
SIM_PARAMS = {
    "dimensions": [1000, 1000, 500],  # LxWxH (m)
    "min_bound": 0.0,
    "max_bound": 1000.0,
    "N_uavs": 4,                     # Số lượng UAV trong đàn
    "N_waypoints": 10,               # Số điểm tham chiếu cho mỗi UAV (M)
    "weights": [1.0, 50.0, 10.0],    # Trọng số [f1: Thời gian, f2: Va chạm, f3: Nhiệm vụ]
}

# --- Dữ liệu Chướng ngại vật Tĩnh (Scenario 1) ---
# Format: [x_center, y_center, z_center, radius_safe_zone]
STATIC_OBSTACLES = [
    [200, 200, 100, 50],
    [750, 800, 250, 70],
    [100, 900, 300, 40],
    [900, 100, 150, 60]
]

# --- Điểm Bắt đầu và Điểm Nhiệm vụ ---
# UAVs đều bắt đầu ở góc dưới và cần đi qua 3 khu vực nhiệm vụ (mission targets)
START_POSITIONS = [
    [10, 10, 50], [10, 20, 50], [20, 10, 50], [20, 20, 50]
]

MISSION_TARGETS = [
    [500, 500, 200, 30], # (x, y, z, required_radius)
    [800, 200, 350, 30],
    [200, 800, 150, 30]
]

# --- Tham số QISO ---
QISO_PARAMS = {
    "N_particles": 200,              # Kích thước quần thể (phải lớn để tận dụng GPU)
    "max_iter": 500,                 # Số lần lặp
    "C1": 1.5,                       # Trọng số P_best
    "C2": 1.5,                       # Trọng số G_best
    "L_factor": 0.5,                 # Hệ số dịch chuyển lượng tử
    "simulation_name": "Scenario_1_Static"
}

# Kết hợp tất cả để dễ dàng truyền vào lớp Environment
SCENARIO_CONFIG = {
    "sim_params": SIM_PARAMS,
    "obstacles_data": STATIC_OBSTACLES,
    "start_pos": START_POSITIONS,
    "mission_targets": MISSION_TARGETS
}
