import cv2
import numpy as np

# 读取深度图
# depth_image = cv2.imread('/home/lbycdy/work/datasets/OCID-VLG/ARID20/table/top/seq13/depth/result_2018-08-23-09-44-51.png', cv2.IMREAD_UNCHANGED)
depth_image = cv2.imread('/home/lbycdy/work/datasets/OCID-VLG/ARID20/table/bottom/seq07/depth/result_2018-08-21-14-07-32.png', cv2.IMREAD_UNCHANGED)

# 归一化深度图到0-255范围
normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

# 使用OpenCV的"海洋"色图，它会生成从蓝色到绿色的渐变
depth_colored = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_OCEAN)

# 显示蓝色调的深度图
cv2.imshow("Ocean Colored Depth", depth_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()
