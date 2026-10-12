import os
import json
import numpy as np

def ray_intersects_aabb(p_start, p_end, aabb_min, aabb_max):
    """
    经典 3D 射线-轴对齐包围盒 (AABB) 碰撞检测算法 (Slab Method)
    检测线段 [p_start, p_end] 是否与包围盒 [aabb_min, aabb_max] 发生几何干涉
    """
    d = p_end - p_start
    t_min = 0.0
    t_max = 1.0

    for i in range(3):
        if abs(d[i]) < 1e-6:
            # 射线与该轴平行
            if p_start[i] < aabb_min[i] or p_start[i] > aabb_max[i]:
                return False
        else:
            t1 = (aabb_min[i] - p_start[i]) / d[i]
            t2 = (aabb_max[i] - p_start[i]) / d[i]
            t_near = min(t1, t2)
            t_far = max(t1, t2)

            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)

            if t_min > t_max:
                return False

    return True

class BiopsyTrajectoryPlanner:
    """
    经皮肺穿刺轨迹规划器：负责入针点搜索与直线避障
    """
    def __init__(self, obstacles_json_path):
        with open(obstacles_json_path, 'r', encoding='utf-8') as f:
            self.obstacles = json.load(f)
        print(f"✅ 成功载入 {len(self.obstacles)} 个肋骨障碍物包围盒。")

    def check_collision(self, entry_pt, target_pt):
        """
        检查从 entry_pt 到 target_pt 的直线段是否碰撞任何肋骨
        """
        entry_pt = np.array(entry_pt)
        target_pt = np.array(target_pt)

        for obs in self.obstacles:
            box_min = np.array(obs['min_mm'])
            box_max = np.array(obs['max_mm'])

            # 增加 2mm 的手术安全余量 (Safety Margin)
            safety_margin = 2.0
            box_min -= safety_margin
            box_max += safety_margin

            if ray_intersects_aabb(entry_pt, target_pt, box_min, box_max):
                return True, obs['id']  # 发生碰撞，返回碰撞的肋骨 ID
        return False, None

    def plan_trajectory(self, target_pt, skin_y_coord=None, search_radius=40.0, step=5.0):
        """
        在靶点正前方的体表皮肤区域内网格搜索安全入针点
        target_pt: [x, y, z] mm
        skin_y_coord: 假定体表胸前壁的 Y 坐标 (如果为 None，默认靶点前方 60mm)
        """
        target_pt = np.array(target_pt)
        
        # 胸壁皮肤大致位置（前胸入针）：Y 坐标比靶点靠前约 50-70mm
        if skin_y_coord is None:
            skin_y_coord = target_pt[1] - 60.0

        candidates = []
        # 在靶点上方 X-Z 平面上以圆形/网格区域撒采样点
        x_range = np.arange(target_pt[0] - search_radius, target_pt[0] + search_radius + 1e-3, step)
        z_range = np.arange(target_pt[2] - search_radius, target_pt[2] + search_radius + 1e-3, step)

        total_tested = 0
        collision_free_paths = []

        for x in x_range:
            for z in z_range:
                # 只在搜索半径圆内采样
                dist_xy = np.sqrt((x - target_pt[0])**2 + (z - target_pt[2])**2)
                if dist_xy > search_radius:
                    continue

                entry_pt = np.array([x, skin_y_coord, z])
                total_tested += 1

                is_collision, hit_id = self.check_collision(entry_pt, target_pt)
                path_length = np.linalg.norm(entry_pt - target_pt)

                if not is_collision:
                    collision_free_paths.append({
                        "entry_point": entry_pt.tolist(),
                        "path_length": path_length,
                        "vector": ((target_pt - entry_pt) / path_length).tolist()
                    })

        # 统计提案指标：轨迹可行率
        success_rate = (len(collision_free_paths) / total_tested) * 100 if total_tested > 0 else 0
        print(f"\n================ 避障轨迹规划结果 ================")
        print(f"评估入针点总数: {total_tested}")
        print(f"安全无碰撞入针点数: {len(collision_free_paths)}")
        print(f"📈 候选轨迹安全成功率: {success_rate:.2f}%")

        if len(collision_free_paths) == 0:
            print("❌ 警告：未找到安全直线路径，可能全部被骨骼遮挡！")
            return None

        # 临床准则优选：选择刺入深度最短（创伤最小）的路径
        best_path = min(collision_free_paths, key=lambda p: p["path_length"])
        print(f">> 🏆 最优安全入针点 (Entry Point): {[round(c, 2) for c in best_path['entry_point']]} mm")
        print(f">> 🏆 靶点坐标 (Target Point): {[round(c, 2) for c in target_pt]} mm")
        print(f">> 🏆 最优穿刺深度: {best_path['path_length']:.2f} mm")
        print("==================================================\n")

        return best_path

if __name__ == "__main__":
    # 载入刚才生成的肋骨障碍物
    obs_file = r"F:\Lung-Nodule\data\demo_nodules\rib_obstacles_0037.json"
    
    # 患者 0037 的结节物理坐标
    target_0037 = [407.46, 288.58, -163.50]

    planner = BiopsyTrajectoryPlanner(obs_file)
    best_trajectory = planner.plan_trajectory(target_0037)

    # 保存最优规划结果供下一阶段机械臂使用
    if best_trajectory is not None:
        out_plan_file = r"F:\Lung-Nodule\data\demo_nodules\planned_trajectory_0037.json"
        with open(out_plan_file, "w", encoding="utf-8") as f:
            json.dump({
                "target_point": target_0037,
                "best_trajectory": best_trajectory
            }, f, indent=4)
        print(f"🎉 规划路径已保存至: {out_plan_file}")