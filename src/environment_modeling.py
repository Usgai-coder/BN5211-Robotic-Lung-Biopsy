import os
import json
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

class VirtualChestEnvironment:
    """
    虚拟胸腔几何建模器：
    从 3D CT 中提取体表皮肤曲面与肋骨骨骼障碍物（Bounding Boxes）
    """
    def __init__(self, ct_path):
        """
        自适应加载患者 CT：
        - 如果是文件夹，自动读取里面的 DICOM 序列 (.dcm)
        - 如果是文件，直接读取 (.mhd / .nii)
        """
        print(f"正在加载 CT 影像，请稍候: {ct_path}")
        if os.path.isdir(ct_path):
            # 处理 DICOM 序列文件夹
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(ct_path)
            reader.SetFileNames(dicom_names)
            self.ct_image = reader.Execute()
        else:
            # 处理单文件
            self.ct_image = sitk.ReadImage(ct_path)

        self.spacing = np.array(self.ct_image.GetSpacing())      # (dx, dy, dz) mm
        self.origin = np.array(self.ct_image.GetOrigin())        # (ox, oy, oz) mm
        self.direction = np.array(self.ct_image.GetDirection()).reshape(3, 3)
        
        # 转为 numpy 数组: 形状通常为 [Z, Y, X]
        self.ct_array = sitk.GetArrayFromImage(self.ct_image)
        print(f"CT 加载完成，原图形状 [Z, Y, X]: {self.ct_array.shape}")

    def voxel_to_world(self, voxel_coord):
        """
        体素坐标 (x_idx, y_idx, z_idx) 转换为绝对物理坐标 (X_mm, Y_mm, Z_mm)
        """
        return self.ct_image.TransformIndexToPhysicalPoint([int(c) for c in voxel_coord])

    def world_to_voxel(self, world_coord):
        """
        绝对物理坐标 (X_mm, Y_mm, Z_mm) 转换为体素索引 (x_idx, y_idx, z_idx)
        """
        return self.ct_image.TransformPhysicalPointToIndex(world_coord)

    def extract_skin_surface(self, skin_hu_threshold=-200):
        """
        提取胸壁皮肤表面：
        通过阈值分割识别整个人体区域，提取最外层边界轮廓点
        """
        print(f"正在提取体表皮肤掩膜 (HU > {skin_hu_threshold})...")
        # 二值化：将体表组织与空气背景分离
        body_mask = (self.ct_array > skin_hu_threshold).astype(np.uint8)
        
        # 填充体腔内部空洞，得到实心人体 mask
        # 针对每个 2D slice 进行二值填充
        for z in range(body_mask.shape[0]):
            body_mask[z] = ndimage.binary_fill_holes(body_mask[z])
            
        return body_mask

    def extract_rib_bounding_boxes(self, bone_hu_threshold=300, min_volume_voxels=300):
        """
        提取肋骨障碍物三维边界框 (Bounding Boxes):
        通过骨骼 HU 阈值分割提取高密度肋骨，按连通域划分，输出三维轴对齐包围盒 (AABB)
        """
        print(f"正在提取骨骼掩膜并计算连通域包围盒 (HU > {bone_hu_threshold})...")
        bone_mask = self.ct_array > bone_hu_threshold
        
        # 连通域标记 (3D Connected Component Labeling)
        labeled_bones, num_features = ndimage.label(bone_mask)
        
        rib_boxes = []
        # 获取各连通域的切片边界
        objects = ndimage.find_objects(labeled_bones)
        
        for idx, loc in enumerate(objects):
            if loc is None:
                continue
            
            # 计算该连通域的体素体积，过滤噪声微小骨片
            voxel_count = np.sum(labeled_bones[loc] == (idx + 1))
            if voxel_count < min_volume_voxels:
                continue
            
            z_slice, y_slice, x_slice = loc
            
            # 体素边界 [min, max]
            # SimpleITK 的坐标转换接受的是 [X, Y, Z] 顺序的索引
            min_voxel = (x_slice.start, y_slice.start, z_slice.start)
            max_voxel = (x_slice.stop - 1, y_slice.stop - 1, z_slice.stop - 1)
            
            # 转换为物理世界坐标边界 (mm)
            min_world = np.array(self.voxel_to_world(min_voxel))
            max_world = np.array(self.voxel_to_world(max_voxel))
            
            # 轴对齐三维包围盒 (AABB): [min_x, min_y, min_z, max_x, max_y, max_z]
            box_world = {
                "id": f"rib_cluster_{idx+1}",
                "min_mm": np.minimum(min_world, max_world).tolist(),
                "max_mm": np.maximum(min_world, max_world).tolist(),
                "volume_voxels": int(voxel_count)
            }
            rib_boxes.append(box_world)
            
        print(f"✅ 成功提取到 {len(rib_boxes)} 个主要骨骼/肋骨障碍物包围盒。")
        return rib_boxes

# ==========================================
# 本地测试与 JSON 导出
# ==========================================
if __name__ == "__main__":
    # 0037 患者的真实 DICOM 序列文件夹路径
    sample_ct_path = r"F:\Lung-Nodule\data\LIDC-IDRI\LIDC-IDRI-0037\1.3.6.1.4.1.14519.5.2.1.6279.6001.250770014904528873190814943829\1.3.6.1.4.1.14519.5.2.1.6279.6001.219909753224298157409438012179"
    
    if not os.path.exists(sample_ct_path):
        print(f"❌ 找不到文件夹：{sample_ct_path}")
    else:
        # 1. 初始化虚拟胸腔环境建模器
        env = VirtualChestEnvironment(sample_ct_path)
        
        # 2. 提取并计算肋骨障碍物 (骨骼 HU > 300，过滤体积小于 300 体素的噪声)
        rib_boxes = env.extract_rib_bounding_boxes(bone_hu_threshold=300, min_volume_voxels=300)
        
        # 3. 严格使用绝对路径保存
        output_dir = r"F:\Lung-Nodule\data\demo_nodules"
        os.makedirs(output_dir, exist_ok=True)
        
        out_path = os.path.join(output_dir, "rib_obstacles_0037.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rib_boxes, f, indent=4)
            
        print(f"🎉 障碍物三维包围盒已成功提取并保存至: {out_path}")