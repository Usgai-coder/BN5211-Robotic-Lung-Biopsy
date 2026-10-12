# 🫁 肺结节 3D 影像标准化预处理与多维特征构建系统 (LIDC-IDRI / LUNA16)

---

## 1. 项目背景与核心学术贡献 (Project Highlights)

本项目致力于解决肺癌早期筛查中人工阅片耗时长、微小病灶易漏诊的临床痛点。基于 **LIDC-IDRI** 与 **LUNA16** 双数据集，构建了从原始异构医学影像（DICOM/MHD）清洗、特征提取、深度学习建模到临床辅助决策的端到端全栈流水线。

- **多流派网络架构矩阵**：完整复现并横向对比了 2D (ResNet, DenseNet) 与 3D (ResNet, VNet) 医学图像分类基线，深度论证了空间立体特征在结节鉴别中的不可替代性。
- **混合注意力机制与消融验证**：自主研发并注入 2D/3D Attention 模块（如 3D CBAM），并通过严谨的消融实验（Ablation Study）证实了其在微小病灶特征捕捉上的显著增益。
- **打破黑盒与多维临床压测**：集成 3D Grad-CAM 逆向投影技术，精准定位良性“中心钙化”与恶性“边缘浸润”特征；并执行了极其严苛的临床亚组分析（Clinical Subgroups）与零样本泛化测试（LUNA16 Generalization）。
- **循证医学应用落地**：基于 Youden Index 计算出最佳临床防漏诊阈值，融合模型集成策略（Ensemble），最终交付高可用性的 Streamlit 靶向药辅助决策 Web 平台。
- **端到端自主机器人穿刺辅助系统 (Cyber-Physical System)**：扩展构建了医疗信息物理闭环，从胸部 CT 提取胸壁与肋骨 AABB 避障环境，利用射线碰撞检测（Ray-AABB）规划最优肋间隙路径，解算 UR5 6-DOF 机械臂逆运动学，并提供高保真 3D 静态与实时动态穿刺仿真。

---

## 2. 核心技术栈 (Tech Stack)

-   **核心驱动语言**: `Python 3.9+`
-   **医疗影像解析底座**:
    *   `SimpleITK`: 支撑三维空间插值、形态学滤波网络及全维度矩阵仿射变换。
    *   `pydicom`: 极速剥离 DICOM 头部信息并提取脱敏临床元数据。
-   **科学计算与特征工程聚合**:
    *   `pandas`: 构建临床宏观账本、执行高维数据聚合与阵营级联筛选。
    *   `numpy`: 驱动三维空间距离运算、HU 标量截断及安全掩膜填充。
-   **深度学习框架**: `PyTorch`, `PyTorch Lightning`
-   **机器人运动学与空间几何**: 基于标准 DH 参数构建的 6-DOF 运动学求解器、Ray-AABB 射线碰撞检测（Slab Method）
-   **多维可视化与动态交互**: `Matplotlib 3D`, `Streamlit`, `Pillow (GIF 动图渲染)`

---

## 3. 项目标准目录结构 (Project Directory Structure)
本仓库的代码架构严格遵循高内聚、低耦合的机器学习工程规范：

```text
Lung-Nodule/
├── data/
│   ├── processed/                       # 数据资产核心库
│   │   ├── all_roi_cubes/               # LIDC 32x32x32 标准化结节张量
│   │   ├── luna16_roi_cubes/            # LUNA16 跨域泛化测试张量
│   │   ├── preprocessed_cubes/          # LUNA16 全肺闭运算预处理矩阵
│   │   ├── cleaned_nodules_labels.csv   # 黄金结节清洗清单 (Ground Truth)
│   │   ├── final_split_train_manifest.csv # 7:1:2 患者级防泄露切分索引
│   │   └── *manifest.csv                # 各阶段资产总账本
│   └── demo_nodules/                    # 手术环境描述、规划轨迹与三维仿真导出物
│
├── models/                              # 模型权重与评估产物
│   ├── checkpoints_*/                   # 细分模型权重库 (2d, 3d, vnet, attention, densenet, ablation)
│   ├── evaluation_results/              # 各类 ROC 曲线、混淆矩阵、统计分析图表
│   └── logs/                            # TensorBoard 训练日志
│
├── src/                                 # 核心闭环源码
│   ├── 预处理与数据管道 (Data Pipeline)
│   │   ├── preprocess.py / explore_data.py / extract_roi.py
│   │   ├── extract_luna16.py            # LUNA16 盲测集独立提取器
│   │   ├── split_dataset.py             # 患者级防穿越数据切分
│   │   └── dataset_2D.py / dataset_3D.py # PyTorch Dataset 加载器
│   │
│   ├── 核心网络架构 (Network Architectures)
│   │   └── model_*.py                   # 涵盖 2D, 3D, DenseNet, VNet 等骨干网络定义
│   │
│   ├── 训练引擎 (Training Modules)
│   │   └── train_*.py                   # 针对不同架构的独立训练脚本 (含 Attention 与 Ablation)
│   │
│   ├── 评估与可解释性 (Evaluation & Interpretability)
│   │   ├── evaluate_*.py                # 多维评估 (2D/3D对比, Subgroups, 泛化测试, Ensemble, 临床阈值)
│   │   └── generate_3D_gradcam.py       # 3D 梯度类激活热力图反演
│   │
│   ├── 手术环境建模与轨迹规划 (Environment & Trajectory Planning)
│   │   ├── environment_modeling.py      # CT 骨骼与胸壁包围盒 (AABB) 提取
│   │   └── trajectory_planner.py        # 基于射线-AABB 碰撞检测的最优路径规划
│   │
│   ├── 机器人运动控制与多模态仿真 (Robotics & Simulation)
│   │   ├── robot_kinematics.py          # UR5 机械臂 DH 建模与逆运动学数值求解
│   │   ├── main_pipeline.py             # 端到端 AI+机器人穿刺总控主流程
│   │   ├── visualize_simulation.py      # 3D 静态手术场景仿真与高清图导出
│   │   └── simulate_dynamic_puncture.py # 动态进针穿刺模拟与演示 GIF 导出
│   │
│   └── 部署与交互 (Deployment)
│       ├── inference_api.py             # 工业级推理张量解析接口
│       └── app.py                       # 👨‍⚕️ Streamlit 临床决策 Web 原型
│
└── requirements.txt                     # 环境依赖清单
```

---

## 4. 全链路运行指南 (Execution Guide)
为确保整个数据流动管道平滑闭环且表头无污染，请严格按照以下一键通关顺序执行：

**阶段一**：数据清洗与物理空间重构
请依次执行以下脚本，完成异构数据的提纯、聚类与 3D ROI 提取：
```text
python src/preprocess.py
python src/explore_data.py
python src/extract_roi.py
python src/extract_luna16.py
python src/split_dataset.py
```

**阶段二**：算法矩阵训练与消融实验
项目支持多维度网络架构的训练对比，可根据研究目的运行对应脚本：
* 基线模型训练对比
```text
python src/train_2D.py
python src/train_3D.py
python src/train_3D_vnet.py
```

* 注意力机制与消融实验
```text
python src/train_3D_attention.py
python src/train_3D_ablation.py
```

**阶段三**：多维临床压测与可解释性反演
模型训练完成后，执行全面的科学评估与视觉映射：
* 基础性能与盲测泛化评估
```text
python src/evaluate_3D_models.py
python src/evaluate_luna16_generalization.py
```

* 临床亚组分析与模型融合
```text
python src/evaluate_clinical_subgroups.py
python src/evaluate_ensemble.py
```

* 打破黑盒：生成 3D 浸润热力图
```text
python src/generate_3D_gradcam.py
```

**阶段四**：寻找最佳临床工作点与 Web 部署
计算最优敏感/特异性平衡点，并启动交互式诊疗系统：
* 锁定 Youden Index 最佳阈值
```text
python src/evaluate_clinical_value.py
```

* 启动 Web 前端交互系统
```text
streamlit run src/app.py
```

**阶段五**：自主机器人穿刺辅助与端到端闭环 
将深度学习临床决策接入虚拟手术环境与机械臂执行终端：
* 提取解剖避障约束
```text
python src/environment_modeling.py
```

* 运行端到端全自动手术流水线 (AI 恶性分层 -> 射线避障规划 -> UR5 逆运动学求解)
```text
python src/main_pipeline.py
```

* 生成 3D 静态手术场景仿真大图 
```text
python src/visualize_simulation.py
```

* 启动动态实时进针仿真并导出动画 
```text
python src/simulate_dynamic_puncture.py
```

---

## 5. 快速开始与环境配置 (Quick Start)

**1**. 创建并激活纯净虚拟环境
```text
conda create -n lung_nodule python=3.9 -y
conda activate lung_nodule
```

**2**. 安装核心依赖
```text
pip install -r requirements.txt
```

---

## 6. 免责声明 (Disclaimer)
本项目模型及 Web 系统输出的恶性概率与风险分层结果，仅供生物医学工程领域的学术研究、算法验证及演示使用。在任何情况下，均不可替代专业注册放射科医师与肿瘤科医师在真实医疗环境中的最终确诊意见。规划轨迹与机械臂控制仿真代码亦不可直接用于真实人体手术操作。
