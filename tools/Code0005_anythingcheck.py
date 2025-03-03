import os
import cv2
import numpy as np

# 设置文件夹路径
parent_folder = '/home/lbycdy/work/datasets/OCID-VLG/ARID20/table/bottom/seq09'
rgb_folder = os.path.join(parent_folder, 'rgb')
depth_folder = os.path.join(parent_folder, 'depth')
mask_folder = os.path.join(parent_folder, 'seg_mask_instances_combi')

# 获取文件列表，仅获取.png文件
rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')])
depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

# 检查每个RGB文件是否有对应的Depth和Mask文件
for rgb_file in rgb_files:
    # 构造对应的Depth和Mask文件名
    depth_file = rgb_file
    mask_file = rgb_file

    if depth_file not in depth_files:
        print(f"Warning: No corresponding Depth file for {rgb_file}")
        continue
    if mask_file not in mask_files:
        print(f"Warning: No corresponding Mask file for {rgb_file}")
        continue

    # 加载图像
    rgb_path = os.path.join(rgb_folder, rgb_file)
    depth_path = os.path.join(depth_folder, depth_file)
    mask_path = os.path.join(mask_folder, mask_file)

    rgb_img = cv2.imread(rgb_path)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 深度图读取为灰度图或16位图
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 掩码图读取为灰度图

    # 检查图像是否加载成功
    if rgb_img is None:
        print(f"Error: Failed to load RGB image: {rgb_file}")
        continue
    if depth_img is None:
        print(f"Error: Failed to load Depth image: {depth_file}")
        continue
    if mask_img is None:
        print(f"Error: Failed to load Mask image: {mask_file}")
        continue

    # 检查图像大小是否一致
    if rgb_img.shape[:2] != depth_img.shape[:2] or rgb_img.shape[:2] != mask_img.shape[:2]:
        print(f"Warning: Image size mismatch! RGB: {rgb_img.shape}, Depth: {depth_img.shape}, Mask: {mask_img.shape}")
        continue

    # 可视化RGB图像
    cv2.imshow('RGB Image', rgb_img)

    # 深度图使用蓝绿色调色板显示（COLORMAP_OCEAN）
    normalized_depth = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)  # 归一化深度图到0-255
    depth_colored = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_OCEAN)  # 使用COLORMAP_OCEAN
    cv2.imshow('Depth Image (Ocean Colored)', depth_colored)

    # 将掩码实例ID映射到彩色显示
    unique_ids = np.unique(mask_img)  # 获取所有唯一的实例ID
    mask_colored = np.zeros((*mask_img.shape, 3), dtype=np.uint8)  # 初始化彩色掩码图

    for instance_id in unique_ids:
        if instance_id == 0:
            continue  # 跳过背景
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)  # 为每个实例生成随机颜色
        mask_colored[mask_img == instance_id] = color  # 将该实例区域涂成对应颜色

    cv2.imshow('Mask Image (Instance Colored)', mask_colored)

    # 检查掩码覆盖在RGB图像上的效果
    overlay = cv2.addWeighted(rgb_img, 0.7, mask_colored, 0.3, 0)  # 将掩码与RGB图像叠加
    cv2.imshow('RGB with Mask Overlay', overlay)

    # 等待按键以显示下一组
    key = cv2.waitKey(0)
    if key == 27:  # 按ESC退出
        break

cv2.destroyAllWindows()
