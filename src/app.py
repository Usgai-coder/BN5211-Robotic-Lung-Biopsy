import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import io

from inference_api import LungNodulePredictor

# ==========================================
# ⚙️ 系统配置与页面初始化
# ==========================================
st.set_page_config(
    page_title="阿斯利康 - 肺结节辅助诊断与分层系统",
    page_icon="🫁",
    layout="wide"
)

@st.cache_resource
def load_ai_engine():
    try:
        predictor = LungNodulePredictor(checkpoint_dir="models/checkpoints_attention/")
        return predictor
    except Exception as e:
        return None

predictor = load_ai_engine()

# 初始化 Session State，用于保存历史诊断记录
if 'diagnosis_history' not in st.session_state:
    st.session_state.diagnosis_history = []

# ==========================================
# 🎨 侧边栏设计 (控制面板与指南)
# ==========================================
st.sidebar.title("🫁 临床决策支持系统")
st.sidebar.markdown("---")
st.sidebar.info(
    "**核心算法底座**:\n 3D CBAM Attention ResNet\n\n"
    "**最佳工作点 (Youden Index)**: 0.45\n\n"
    "**系统敏感性**: 78% | **系统特异性**: 82%"
)
st.sidebar.markdown(
    "### 💊 靶向药干预指南\n"
    "- **高风险 (>70%)**: 建议基因检测，考虑奥希替尼等靶向药介入。\n"
    "- **中风险 (45%-70%)**: 建议穿刺活检或 3 个月内复查 HRCT。\n"
    "- **低风险 (<45%)**: 建议常规年度低剂量螺旋 CT (LDCT) 随访。"
)

# ==========================================
# 🖥️ 主界面设计 (批量上传与结果展示)
# ==========================================
st.title("胸部 CT 影像肺结节辅助诊断与分层平台")
st.markdown("请上传基于标准化流水线提取的结节 ROI `.npy` 文件 (支持批量多选)。系统将自动调用 3D 卷积引擎进行空间特征解析、风险分层并生成报告。")

if predictor is None:
    st.error("❌ 模型加载失败！请检查 `models/checkpoints_attention/` 目录下是否存在有效的权重文件。")
    st.stop()

# 🚀 改进一：支持 accept_multiple_files=True 批量上传
uploaded_files = st.file_uploader("📂 批量上传结节 ROI 数据 (.npy)", type=["npy"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"✅ 成功接收 {len(uploaded_files)} 个病灶数据，正在启动并行推理...")
    
    # 清空上次的临时列表（但保留 session_state 中的全局历史，如果需要的话）
    current_batch_results = []
    
    # 建立多列布局
    col1, col2 = st.columns([1, 1.5])
    
    for file in uploaded_files:
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        
        try:
            # 执行推理
            result = predictor.predict_single(temp_path)
            prob_percent = result['malignancy_probability'] * 100
            
            # 判断临床分级
            if prob_percent < 45.0:
                risk_level = "低风险 (常规随访)"
                risk_emoji = "🟢"
            elif 45.0 <= prob_percent <= 70.0:
                risk_level = "中风险 (建议活检/短随访)"
                risk_emoji = "🟡"
            else:
                risk_level = "高风险 (极高疑似，建议靶向药筛查)"
                risk_emoji = "🔴"
                
            # 记录数据
            record = {
                "结节编号": file.name,
                "AI推断恶性概率": f"{prob_percent:.2f}%",
                "临床风险分级": risk_level,
                "建议操作": risk_emoji
            }
            current_batch_results.append(record)
            st.session_state.diagnosis_history.append(record)
            
            # 在左侧渲染最后一个图像的预览（如果文件太多，全画出来会卡，画最后一个作为抽样预览）
            with col1:
                if file == uploaded_files[-1]:
                    st.subheader(f"👁️ 样本切片预览: {file.name}")
                    cube = np.load(temp_path)
                    if cube.shape == (32, 32, 32):
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(cube[16, :, :], cmap='bone')
                        ax.axis('off')
                        st.pyplot(fig)
            
            # 清理临时文件
            os.remove(temp_path)
            
        except Exception as e:
            st.error(f"处理 {file.name} 时发生错误: {e}")

    # 🚀 改进二：实时汇总表格展示
    with col2:
        st.subheader("📊 批量诊断分析汇总表")
        df_results = pd.DataFrame(current_batch_results)
        # 将结果展示为漂亮的交互式 Dataframe
        st.dataframe(df_results, use_container_width=True)
        
        # 🚀 改进三：一键导出 CSV 诊断报告
        csv_buffer = io.BytesIO()
        df_results.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="💾 导出本次诊断报告 (CSV)",
            data=csv_data,
            file_name="肺结节智能辅助诊断报告.csv",
            mime="text/csv",
            type="primary"
        )

st.markdown("---")
st.caption("⚖️ 免责声明：本交互式原型系统仅用于学术演示与科研验证，其推断概率基于 LIDC-IDRI 历史数据集分布，不作为真实医疗环境中的最终确诊依据。")