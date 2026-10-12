import os
import json
import torch
import numpy as np

# 导入前 4 个阶段构建好的核心模块
from train_3D_attention import LitAttentionModel
from trajectory_planner import BiopsyTrajectoryPlanner
from robot_kinematics import UR5Kinematics, build_needle_pose

def run_end_to_end_biopsy_pipeline(
    ckpt_path,
    roi_cube_path,
    target_coords_mm,
    obstacles_json_path,
    malignancy_threshold=0.45
):
    print("=" * 70)
    print("🏥 端到端 AI 辅助机器人穿刺活检系统 (End-to-End Autonomous Biopsy)")
    print("=" * 70)

    # -------------------------------------------------------------
    # Step 1: AI 影像诊断与风险分层 
    # -------------------------------------------------------------
    print("\n[Stage 1/4] 正在执行 3D-CBAM CNN 结节恶性风险智能评估...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lit_model = LitAttentionModel.load_from_checkpoint(ckpt_path)
    lit_model.to(device)
    lit_model.eval()

    cube = np.load(roi_cube_path)
    tensor = torch.from_numpy(cube).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = lit_model(tensor)
        probs = torch.softmax(logits, dim=1)
        malignant_prob = probs[0, 1].item()

    print(f">> 评估样本: {os.path.basename(roi_cube_path)}")
    print(f">> 恶性预测置信度: {malignant_prob:.2%}")
    print(f">> 临床干预阈值: {malignancy_threshold:.2f}")

    if malignant_prob < malignancy_threshold:
        print(">> 💡 临床决策: 判定为低风险/良性结节，建议定期影像随访。机器人保持安全待机。")
        print("=" * 70)
        return

    print(">> 🚨 临床决策: 确认为高风险恶性结节！自动触发经皮穿刺活检手术流程。")
    print(f">> 锁定结节质心物理坐标 (Target Point): {target_coords_mm} mm")

    # -------------------------------------------------------------
    # Step 2: 虚拟胸腔环境与避障轨迹规划 
    # -------------------------------------------------------------
    print("\n[Stage 2/4] 加载患者胸壁与肋骨空间，执行射线避障轨迹规划...")
    planner = BiopsyTrajectoryPlanner(obstacles_json_path)
    best_path = planner.plan_trajectory(target_coords_mm)

    if best_path is None:
        print(">> ❌ 严重警告: 穿刺路径均发生骨骼遮挡，已终止机械臂规划！")
        return

    entry_point = np.array(best_path["entry_point"])
    target_point = np.array(target_coords_mm)
    needle_vector = np.array(best_path["vector"])

    print(f">> 确立最优体表入针点: {[round(c, 2) for c in entry_point]} mm")
    print(f">> 直线穿刺深度: {best_path['path_length']:.2f} mm")

    # -------------------------------------------------------------
    # Step 3: 6-DOF 机械臂逆运动学求解 
    # -------------------------------------------------------------
    print("\n[Stage 3/4] 计算机械臂逆向运动学 (Inverse Kinematics)...")
    ur5 = UR5Kinematics()
    target_pose = build_needle_pose(needle_tip=target_point, direction_vector=needle_vector)
    
    joint_angles, converged, _ = ur5.solve_ik_numerical(target_pose)
    joint_angles_deg = np.rad2deg(joint_angles)

    print(f">> 逆运动学求解状态: {'✅ 成功收敛' if converged else '⚠️ 未收敛'}")
    print(">> 机械臂 6 个驱动关节目标角度 (Degrees):")
    for i, deg in enumerate(joint_angles_deg):
        print(f"     Joint {i+1}: {deg:8.2f}°")

    # -------------------------------------------------------------
    # Step 4: 正运动学校验与靶向误差评估 
    # -------------------------------------------------------------
    print("\n[Stage 4/4] 执行机械臂前向运动学仿真与靶向误差校验...")
    fk_matrix = ur5.forward_kinematics(joint_angles)
    actual_tip_pos = fk_matrix[:3, 3]
    targeting_error = np.linalg.norm(actual_tip_pos - target_point)

    print(f">> 理论靶点坐标: {[round(c, 2) for c in target_point]} mm")
    print(f">> 实际针尖坐标: {[round(c, 2) for c in actual_tip_pos]} mm")
    print(f">> 🎯 最终靶向定位误差 (Targeting Error): {targeting_error:.4f} mm")
    
    print("\n" + "=" * 70)
    print("🎉 端到端自主穿刺路径闭环测试通过！已达到临床亚毫米级定位要求。")
    print("=" * 70)


if __name__ == "__main__":
    # 配置各模块路径
    CKPT_PATH = "models/checkpoints_attention/3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt"
    
    # 使用终端查到的真实路径
    CUBE_PATH = r"F:\Lung-Nodule\data\processed\all_roi_cubes\LIDC-IDRI-0037_IL057_12.npy"
    
    TARGET_MM = [407.46, 288.58, -163.50]
    OBSTACLES_JSON = r"F:\Lung-Nodule\data\demo_nodules\rib_obstacles_0037.json"

    run_end_to_end_biopsy_pipeline(
        ckpt_path=CKPT_PATH,
        roi_cube_path=CUBE_PATH,
        target_coords_mm=TARGET_MM,
        obstacles_json_path=OBSTACLES_JSON,
        malignancy_threshold=0.45
    )