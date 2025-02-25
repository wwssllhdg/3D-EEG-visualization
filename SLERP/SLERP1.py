import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
import pyvista as pv

combined_mesh = trimesh.load_mesh("../Dataset/combined_brain.obj")
vertices = combined_mesh.vertices

# 加载功率数据
power_data = np.load('../Wavelet Transform/power_data.npy')  # 形状 (21, 5, 12000)

# 假设电极位置
electrode_positions = np.array([
    (-18.359825, 122.168060, -12.120007), #Fp1
    (11.40872, 118.90113, -12.3000065),   #Fpz
    (41.177265, 115.634201, -12.480006),  #Fp2
    (-61.964268, 82.256561, -16.200003),  #F7
    (-37.271305, 98.767265, 26.880000),   #F3
    (4.994864, 84.830490, 52.680000),     #Fz
    (53.034767, 88.698257, 26.519993),    #F4
    (70.882820, 66.899742, -16.920002),   #F8
    (-86.147141, 17.879263, -24.840000),  #T3
    (-64.116859, 21.854326, 57.360001),   #C3
    (-0.581079, 18.343462, 76.680000),    #Cz
    (62.790398, 8.824049, 57.000000),     #C4
    (83.316093, -1.644724, -21.840000),   #T4
    (-80.348694, -35.629734, 4.560006),   #T5
    (-55.199612, -51.080410, 44.880013),  #P3
    (-6.255757, -38.257339, 65.400009),   #Pz
    (36.158401, -57.386772, 47.040005),   #P4
    (63.819141, -51.013313, 4.560006),    #T6
    (-42.517223, -80.979530, 3.360009),   #O1
    (-8.366949, -59.634697, 6.000007),    #Oz
    (18.298971, -86.661629, 4.920010),    #O2
])

# 计算球面插值权重矩阵
#计算了脑皮层表面各个点到每个电极的加权值，用于后续的插值
electrode_norm = electrode_positions / np.linalg.norm(electrode_positions, axis=1, keepdims=True)
vertices_norm = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

cos_theta = np.dot(vertices_norm, electrode_norm.T)
cos_theta = np.clip(cos_theta, -1.0, 1.0)
theta = np.arccos(cos_theta)

epsilon = 1e-6
weights = 1.0 / (theta**2 + epsilon)
weights /= weights.sum(axis=1, keepdims=True)

# 输出权重矩阵
print("权重矩阵：")
print(weights)
weights = weights.astype(np.float32)


# 插值功率数据并保存矩阵
num_points = 265396
num_timepoints = 200
num_bands = 5
frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

# 生成5个矩阵（一个频段对应一个矩阵）
output_directory = "../Dataset/Different frequency power matrices"

for freq_index in range(num_bands):
    electrode_power = power_data[:, freq_index, :]
    electrode_power = electrode_power.astype(np.float32)
    interpolated_data = np.dot(weights, electrode_power)
    min_p = interpolated_data.min(axis=0)
    max_p = interpolated_data.max(axis=0)
    normalized_data = (interpolated_data - min_p) / (max_p - min_p + 1e-6)
    print(freq_index)
    # 生成完整的文件路径
    file_path = os.path.join(output_directory, f'{frequency_bands[freq_index]}.npy')
    # 保存数据到指定路径
    np.save(file_path, normalized_data)
    print(f"数据已保存到: {file_path}")


# 1. 创建多边形网格并输出调试信息
print("Creating the mesh...")
faces_pv = np.hstack([np.full((combined_mesh.faces.shape[0], 1), 3), combined_mesh.faces]).astype(np.int64)
mesh = pv.PolyData(vertices, faces_pv)
print(f"Mesh created with {mesh.n_points} points and {mesh.n_faces} faces.")

# 2. 为网格添加 'power' 数据
print("Adding power data to mesh...")
mesh.point_data['power'] = normalized_data[:, 0]
print(f"Power data added for first time point (shape: {normalized_data[:, 0].shape}).")

# 3. 初始化可视化器并输出调试信息
print("Initializing the plotter...")
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='power', cmap='jet', opacity='linear', clim=[0, 1])
print("Mesh added to plotter. Showing initial plot...")
plotter.show(auto_close=False)

# 4. 动态更新并调试输出
for t in range(normalized_data.shape[1]):
    print(f"Updating time point {t+1}/{normalized_data.shape[1]}...")
    mesh.point_data['power'] = normalized_data[:, t]
    plotter.update_scalars(mesh['power'], render=False)
    plotter.render()
    plt.pause(0.01)
    print(f"Time point {t+1} rendered.")

# 5. 结束并关闭图形
print("Rendering complete. Closing plotter...")
plotter.close()
print("Plotter closed. Process finished.")