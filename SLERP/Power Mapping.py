import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 假设文件名为 delta.npy, theta.npy, alpha.npy, beta.npy, gamma.npy
frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
color_weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # 可以设置不同频段的权重

# 用于存储五个频段的矩阵
power_band_matrices = []

# 读取每个频段的矩阵文件
for band in frequency_bands:
    file_path = f'../Dataset/Different frequency power matrices/{band}.npy'  # 假设文件位于当前目录
    power_band_data = np.load(file_path)
    power_band_matrices.append(power_band_data)
    print(f"读取 {band} 的数据，形状: {power_band_data.shape}")

# power_band_matrices 中现在包含了五个矩阵，每个矩阵的形状是 (265396, 200)

# 颜色映射：你可以选择任何合适的 colormap，使用 matplotlib 的 colormap
cmap = plt.get_cmap('jet')  # 使用 jet 色图

# 1. 为每个矩阵创建颜色和不透明度映射
color_mappings = []  # 用于保存颜色映射
opacity_mappings = []  # 用于保存不透明度映射

# 2. 对于每个频段进行处理，计算加权颜色和透明度
for idx, matrix in enumerate(power_band_matrices):
    print(f"处理第 {idx + 1} 个频段的矩阵...")

    # 将矩阵的每个值归一化到[0, 1]范围，用于颜色和不透明度的映射
    min_val = matrix.min()
    max_val = matrix.max()
    normalized_matrix = (matrix - min_val) / (max_val - min_val + 1e-6)  # 避免除零错误

    # 颜色映射：从 colormap 获取 RGB
    # 对于每个时间点，使用归一化的值进行颜色映射
    colors = cmap(normalized_matrix)  # 颜色映射到 RGB 格式 (shape: (265396, 200, 4))

    # 加权颜色：按频段权重加权
    weighted_colors = colors[..., :3] * color_weights[idx]  # 每个频段的RGB值根据权重进行缩放
    color_mappings.append(weighted_colors)

    # 不透明度映射：透明度 = 1 - 归一化功率
    opacity_mapping = 1 - normalized_matrix  # 归一化功率越大，透明度越小
    opacity_mappings.append(opacity_mapping)  # 透明度值在[0, 1]之间

    # 输出调试信息
    print(f"频段 {idx + 1} 的颜色映射形状: {weighted_colors.shape}")
    print(f"频段 {idx + 1} 的不透明度映射形状: {opacity_mapping.shape}")

# 3. 合成最终的颜色和不透明度映射
final_color_mapping = np.sum(color_mappings, axis=0)  # 将所有频段的颜色加权叠加
final_color_mapping = np.clip(final_color_mapping, 0, 1)  # 确保颜色在[0, 1]范围内

# 合成的透明度：计算每个点的平均透明度
final_opacity_mapping = np.mean(opacity_mappings, axis=0)  # 求每个点的透明度平均值

# 4. 保存结果
output_directory = "../Dataset/coloropacity"

# 保存颜色映射
color_mappings_flat = final_color_mapping.reshape(-1, 3)  # 将每个矩阵拉平成一维并堆叠
color_mappings_file = os.path.join(output_directory, 'color_mappings.txt')
np.savetxt(color_mappings_file, color_mappings_flat, fmt='%.6f')

# 保存不透明度映射
opacity_mappings_flat = final_opacity_mapping.flatten()  # 直接将透明度矩阵拉平成一维
opacity_mappings_file = os.path.join(output_directory, 'opacity_mappings.txt')
np.savetxt(opacity_mappings_file, opacity_mappings_flat, fmt='%.6f')

print(f"颜色映射和不透明度映射已经保存为 '{color_mappings_file}' 和 '{opacity_mappings_file}'.")
