import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, random_split

from dataset_3D import LungNodule3DDataset
from train_3D_attention import LitAttentionModel

def get_best_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts: return None
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]

if __name__ == "__main__":
    print("=== 🚀 临床价值验证与最佳阈值 (Optimal Threshold) 分析 ===")
    out_dir = "models/evaluation_results/clinical_value/"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 自动加载最强的 3D Attention 模型
    ckpt_path = get_best_ckpt("models/checkpoints_attention/")
    print(f"📦 加载模型权重: {os.path.basename(ckpt_path)}")
    model = LitAttentionModel.load_from_checkpoint(ckpt_path).eval()

    # 2. 准备纯净的验证集
    MANIFEST_PATH = "data/processed/final_dataset_manifest.csv"
    CUBE_DIR = "data/processed/all_roi_cubes/"
    
    torch.manual_seed(42) # 保证和之前的测试集严格一致
    full_dataset = LungNodule3DDataset(manifest_path=MANIFEST_PATH, cube_dir=CUBE_DIR, transform=None)
    train_size = int(0.8 * len(full_dataset))
    _, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 3. 收集所有的预测概率和真实标签
    all_probs, all_labels = [], []
    print("⏳ 正在验证集上执行全量推理...")
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 4. 计算 ROC 曲线各项参数
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # 5. 核心医学统计学计算：Youden's J statistic (约登指数)
    # 约登指数 = 敏感性 + 特异性 - 1 = TPR - FPR
    # 约登指数最大的点，就是最能平衡“漏诊”和“误诊”的最佳阈值
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    opt_tpr = tpr[optimal_idx]  # 敏感性 (Sensitivity) -> 越高漏诊越少
    opt_fpr = fpr[optimal_idx]  # 1 - 特异性 (1 - Specificity)
    opt_spec = 1 - opt_fpr      # 特异性 -> 越高误诊越少

    # 打印临床核心指标
    print("\n📊 === 临床决策核心指标分析 ===")
    print(f"🔹 总体诊断 AUC   : {roc_auc:.4f}")
    print(f"⭐ 最佳工作点阈值 : {optimal_threshold:.4f} (默认是 0.5000)")
    print(f"📈 敏感性 (Sens)  : {opt_tpr:.4f}  👉 意味着漏诊率为 {1-opt_tpr:.4f}")
    print(f"📉 特异性 (Spec)  : {opt_spec:.4f}  👉 意味着误诊率为 {opt_fpr:.4f}")
    print("===================================\n")

    # 6. 绘制带有最佳工作点的临床级 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Model ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 用红星标注最佳工作点
    plt.scatter(opt_fpr, opt_tpr, color='red', s=120, zorder=5, marker='*', 
                label=f'Optimal Threshold: {optimal_threshold:.2f}')
    
    # 添加数值注释
    plt.annotate(f'Sensitivity: {opt_tpr:.2f}\nSpecificity: {opt_spec:.2f}', 
                 (opt_fpr, opt_tpr), textcoords="offset points", xytext=(15,-15), 
                 ha='left', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.1))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Clinical ROC Curve & Optimal Operating Point', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(out_dir, "clinical_roc_optimal_point.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 临床工作点 ROC 曲线已高清保存至: {plot_path}")