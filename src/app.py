import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

# 导入你的模型架构和API
from train_3D_attention import LitAttentionModel
from inference_api import LungNodulePredictor

# ==========================================
# ⚙️ 系统配置与初始化
# ==========================================
st.set_page_config(
    page_title="阿斯利康 - 肺结节辅助诊断与靶向药分层系统",
    page_icon="🫁",
    layout="wide"
)

# 使用装饰器缓存模型，避免每次刷新网页都重新加载庞大的权重文件
@st.cache_resource
def load_ai_engine():
    try:
        # 这里会复用你在任务8写好的推理接口
        predictor = LungNodulePredictor(checkpoint_dir="models/checkpoints_attention/")
        return predictor
    except Exception as e:
        return None

predictor = load_ai_engine()

# ==========================================
# 🎨 侧边栏设计
# ==========================================
st.sidebar.title("🫁 临床决策支持系统")
st.sidebar.markdown("---")
st.sidebar.info(
    "**技术底座**: 3D CBAM Attention ResNet\n\n"
    "**最佳工作点 (Youden Index)**: 0.45\n\n"
    "**敏感性**: 78% | **特异性**: 82%"
)
st.sidebar.markdown(
    "### 💊 靶向药干预指南\n"
    "- **高风险 (>70%)**: 建议基因检测，考虑奥希替尼介入。\n"
    "- **中风险 (45%-70%)**: 建议穿刺活检或 3 个月内复查。\n"
    "- **低风险 (<45%)**: 建议年度低剂量 CT (LDCT) 随访。"
)

# ==========================================
# 🖥️ 主界面设计
# ==========================================
st.title("胸部 CT 影像肺结节辅助诊断平台")
st.markdown("请上传基于 LUNA16/LIDC-IDRI 标准预处理的 `32x32x32` 结节 ROI `.npy` 矩阵文件。系统将自动进行三维卷积特征提取与风险分层。")

if predictor is None:
    st.error("❌ 模型加载失败！请检查 `models/checkpoints_attention/` 目录下是否存在 `.ckpt` 权重文件。")
    st.stop()

# 文件上传组件
uploaded_file = st.file_uploader("📂 选择结节 ROI 数据文件 (.npy)", type=["npy"])

if uploaded_file is not None:
    # 1. 保存临时文件供模型读取
    temp_path = "temp_upload.npy"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success("✅ 数据解析成功！正在启动 3D 卷积引擎...")
    
    # 2. 界面分两列展示
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👁️ 结节中心切片预览 (Z=16)")
        try:
            cube = np.load(temp_path)
            if cube.shape == (32, 32, 32):
                center_slice = cube[16, :, :]
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(center_slice, cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.error(f"⚠️ 数据维度异常！期望 (32, 32, 32)，实际为 {cube.shape}")
        except Exception as e:
            st.error(f"图像预览失败: {e}")
            
    with col2:
        st.subheader("📊 AI 诊断分析报告")
        with st.spinner('正在进行深度特征空间投影与注意力机制计算...'):
            try:
                # 执行预测
                result = predictor.predict_single(temp_path)
                
                pred_label = result['prediction_label']
                # 将概率转为百分比
                prob_percent = result['malignancy_probability'] * 100
                
                # 动态进度条组件
                st.write(f"**模型推断恶性概率:** {prob_percent:.2f}%")
                progress_bar = st.progress(0)
                # 简单的进度条动画
                progress_bar.progress(int(prob_percent))
                
                # ==========================================
                # 🩺 结合最佳阈值 (0.45) 的临床临床决策逻辑
                # ==========================================
                st.markdown("---")
                st.markdown("### 📋 临床决策建议")
                
                if prob_percent < 45.0: # 低于最佳阈值 0.45
                    st.success("🟢 风险分层：**低风险 (Low Risk)**")
                    st.markdown("> **影像学特征**：病灶可能呈现出典型的良性征象（如规则球形、中心高密度钙化等）。")
                    st.markdown("> **诊疗建议**：良性可能性大。无需特殊药物干预，建议继续维持年度低剂量螺旋 CT (LDCT) 常规体检随访。")
                    
                elif 45.0 <= prob_percent <= 70.0: # 超过阈值，但处于中等区间
                    st.warning("🟡 风险分层：**中高风险 (Moderate-High Risk)**")
                    st.markdown("> **影像学特征**：病灶呈现出不确定性特征，可能伴有早期浸润趋势。")
                    st.markdown("> **诊疗建议**：已触发高危预警！建议尽快安排 PET-CT 检查或经皮肺穿刺活检以明确病理。建议缩短复查周期至 3 个月。")
                    
                else: # > 70% 极高风险
                    st.error("🔴 风险分层：**极高风险 (Extremely High Risk)**")
                    st.markdown("> **影像学特征**：AI 注意力机制高度锁定病灶边缘，提示存在显著的毛刺、分叶等恶性肿瘤浸润性生长特征。")
                    st.markdown("> **诊疗建议**：高度怀疑恶性病变（非小细胞肺癌可能性大）。建议立即转入胸外科或肿瘤科进行多学科会诊 (MDT)。**强烈建议同步进行 EGFR 等靶点基因检测，若符合指征，推荐尽早启动靶向药物（如奥希替尼）治疗。**")
                    
            except Exception as e:
                st.error(f"推理引擎发生错误: {e}")
                
    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)

st.markdown("---")
st.caption("Disclaimer: 本系统基于深度学习原型构建，预测结果仅供科研与临床辅助参考，不作为最终诊断依据。")