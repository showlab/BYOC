
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 假设 df 是从 CSV 文件读取的 DataFrame
df = pd.read_csv(r'E:\Projects\Pycharm-projects\facial_expression_parameter\image2bs\landmark_results\00048.csv')  # 替换为你的CSV文件路径

# 创建一个 224x224 的黑色图像
image = np.zeros((224, 224, 3), dtype=np.uint8)

# 假设 CSV 文件中的坐标已经是相对于 224x224 图像归一化的
# 遍历所有坐标点并在图像上标记
for index, row in df.iterrows():
    x = int(row['X'] * 224)  # 缩放 X 坐标
    y = int(row['Y'] * 224)  # 缩放 Y 坐标
    image[y, x] = [255, 255, 255]  # 设置像素为白色

# 显示图像
plt.imshow(image)
plt.axis('off')  # 不显示坐标轴
plt.show()
