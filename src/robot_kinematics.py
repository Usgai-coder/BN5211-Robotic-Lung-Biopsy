import os
import json
import numpy as np

class UR5Kinematics:
    """
    UR5 6-DOF 机械臂运动学求解器 (基于标准 DH 参数)
    单位：长度 mm，角度 rad
    """
    def __init__(self):
        # UR5 的标准几何尺寸参数 (单位: mm)
        self.d = np.array([89.159, 0.0, 0.0, 109.15, 94.65, 82.3])
        self.a = np.array([0.0, -425.0, -392.25, 0.0, 0.0, 0.0])
        self.alpha = np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])

    def dh_transform(self, a, alpha, d, theta):
        """计算单个连杆的 4x4 齐次变换矩阵"""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0.0,            np.sin(alpha),                np.cos(alpha),               d],
            [0.0,            0.0,                          0.0,                         1.0]
        ])

    def forward_kinematics(self, joint_angles):
        """
        正运动学 (FK)：输入 6 个关节角 (rad)，计算末端执行器 4x4 齐次变换矩阵
        """
        T = np.eye(4)
        for i in range(6):
            T_i = self.dh_transform(self.a[i], self.alpha[i], self.d[i], joint_angles[i])
            T = np.dot(T, T_i)
        return T

    def solve_ik_numerical(self, target_pose, initial_q=None, max_iters=200, tol=1e-3):
        """
        阻尼最小二乘法 (Damped Least Squares / Levenberg-Marquardt) 数值逆运动学求解
        target_pose: 4x4 目标齐次变换矩阵
        """
        if initial_q is None:
            # 默认给一个典型的朝前弯曲工作位姿
            q = np.array([0.0, -np.pi/4, np.pi/2, -np.pi/4, np.pi/2, 0.0])
        else:
            q = np.array(initial_q, dtype=float)

        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]

        for it in range(max_iters):
            T_current = self.forward_kinematics(q)
            current_pos = T_current[:3, 3]
            current_rot = T_current[:3, :3]

            # 位置误差
            pos_err = target_pos - current_pos

            # 姿态误差 (旋转矩阵对数/轴角近似)
            rot_err_mat = np.dot(target_rot, current_rot.T)
            rot_err = 0.5 * np.array([
                rot_err_mat[2, 1] - rot_err_mat[1, 2],
                rot_err_mat[0, 2] - rot_err_mat[2, 0],
                rot_err_mat[1, 0] - rot_err_mat[0, 1]
            ])

            err = np.hstack([pos_err, rot_err * 50.0])  # 权衡位置与姿态权重
            if np.linalg.norm(pos_err) < tol:
                return q, True, np.linalg.norm(pos_err)

            # 数值微分计算雅可比矩阵 (Jacobian)
            J = np.zeros((6, 6))
            delta = 1e-6
            for j in range(6):
                q_plus = q.copy()
                q_plus[j] += delta
                T_plus = self.forward_kinematics(q_plus)
                dp = (T_plus[:3, 3] - current_pos) / delta
                
                # 简单角速度近似
                R_diff = np.dot(T_plus[:3, :3], current_rot.T)
                dw = 0.5 * np.array([
                    R_diff[2, 1] - R_diff[1, 2],
                    R_diff[0, 2] - R_diff[2, 0],
                    R_diff[1, 0] - R_diff[0, 1]
                ]) / delta
                J[:, j] = np.hstack([dp, dw * 50.0])

            # 阻尼最小二乘更新关节角
            damping = 1e-2
            J_dls = np.dot(J.T, np.linalg.inv(np.dot(J, J.T) + damping**2 * np.eye(6)))
            dq = np.dot(J_dls, err)

            q += dq
            # 限制在 [-2pi, 2pi] 范围内
            q = (q + np.pi) % (2 * np.pi) - np.pi

        return q, False, np.linalg.norm(pos_err)


def build_needle_pose(needle_tip, direction_vector):
    """
    根据针尖物理坐标和刺入单位向量，构建末端执行器的 4x4 齐次变换矩阵
    默认设定机械臂末端 Z 轴对齐穿刺向量 direction_vector
    """
    z_axis = np.array(direction_vector)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # 选取一个与 z 轴不平行的辅助向量构造坐标系
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_axis, up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = needle_tip
    return pose


def main():
    print("=== BN5211: 6-DOF 机械臂逆运动学与靶向误差评估 ===")

    # 1. 读取轨迹数据
    trajectory_file = r"F:\Lung-Nodule\data\demo_nodules\planned_trajectory_0037.json"
    if not os.path.exists(trajectory_file):
        print(f"❌ 找不到规划文件: {trajectory_file}")
        return

    with open(trajectory_file, 'r', encoding='utf-8') as f:
        plan_data = json.load(f)

    target_pt = np.array(plan_data["target_point"])
    entry_pt = np.array(plan_data["best_trajectory"]["entry_point"])
    needle_vec = np.array(plan_data["best_trajectory"]["vector"])

    print(f">> 入针点 (Entry Point): {entry_pt}")
    print(f">> 靶点 (Target Point): {target_pt}")
    print(f">> 进针方向向量: {needle_vec}")

    # 2. 构建目标位姿 (当穿刺针尖到达靶点时的期望末端位姿)
    target_pose = build_needle_pose(needle_tip=target_pt, direction_vector=needle_vec)

    # 3. 求解逆运动学 (IK)
    ur5 = UR5Kinematics()
    joint_angles, converged, final_err = ur5.solve_ik_numerical(target_pose)

    joint_angles_deg = np.rad2deg(joint_angles)

    print("\n================ 运动学求解结果 ================")
    print(f"IK 收敛状态: {'✅ 成功收敛' if converged else '⚠️ 未完全收敛'}")
    print("解算出的 6 个关节角度 (Degrees):")
    for i, deg in enumerate(joint_angles_deg):
        print(f"  Joint {i+1} (Theta_{i+1}): {deg:.2f}°")

    # 4. 正运动学校验与计算 Targeting Error (提案第 5 节关键指标)
    fk_result = ur5.forward_kinematics(joint_angles)
    simulated_tip_pos = fk_result[:3, 3]

    # 欧氏距离 (Euclidean Distance)
    targeting_error = np.linalg.norm(simulated_tip_pos - target_pt)

    print("\n================ 提案核心指标评估 ================")
    print(f"理论结节质心坐标: {[round(c, 2) for c in target_pt]} mm")
    print(f"机械臂实际针尖坐标: {[round(c, 2) for c in simulated_tip_pos]} mm")
    print(f"🎯 靶向定位误差 (Targeting Error): {targeting_error:.4f} mm")
    print("==================================================\n")

    # 5. 保存结果供后续集成或可视化使用
    out_kinematics_file = r"F:\Lung-Nodule\data\demo_nodules\kinematics_result_0037.json"
    result_data = {
        "joint_angles_deg": joint_angles_deg.tolist(),
        "joint_angles_rad": joint_angles.tolist(),
        "simulated_tip_pos": simulated_tip_pos.tolist(),
        "theoretical_target": target_pt.tolist(),
        "targeting_error_mm": float(targeting_error),
        "ik_converged": bool(converged)
    }
    with open(out_kinematics_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)
    print(f"🎉 运动学执行结果已保存至: {out_kinematics_file}")


if __name__ == "__main__":
    main()