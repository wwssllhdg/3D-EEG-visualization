import trimesh
import numpy as np
import pyvista as pv
import time

# 加载3D网格
combined_mesh = trimesh.load_mesh("../Dataset/combined_brain.obj")
vertices = combined_mesh.vertices  # 获取顶点坐标
faces = combined_mesh.faces  # 获取面（三角形）索引

# 读取颜色和透明度映射数据
color_file_path = "../Dataset/coloropacity/color_mappings.txt"
opacity_file_path = "../Dataset/coloropacity/opacity_mappings.txt"

# 读取颜色映射（假设颜色映射已经在[0, 1]范围内）
color_mappings = np.loadtxt(color_file_path).reshape(-1, 3)  # 形状为 (顶点数 * 帧数, 3)
opacity_mappings = np.loadtxt(opacity_file_path)  # 形状为 (顶点数 * 帧数,)

# 打印调试信息
print(f"网格顶点数: {vertices.shape[0]}")
print(f"颜色映射的总顶点数: {color_mappings.shape[0]}")
print(f"透明度映射的总顶点数: {opacity_mappings.shape[0]}")

# 确保颜色和透明度的长度与顶点数量和帧数一致
num_frames = color_mappings.shape[0] // vertices.shape[0]  # 计算帧数
assert color_mappings.shape[0] == vertices.shape[0] * num_frames, "颜色映射顶点数与网格顶点数和帧数不匹配"
assert opacity_mappings.shape[0] == vertices.shape[0] * num_frames, "透明度映射顶点数与网格顶点数和帧数不匹配"

# 设置渲染帧数和每帧持续时间
frame_count = 20
frame_duration = 1  # 每帧持续3秒

# 创建 pyvista 的 PolyData 对象
faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
mesh = pv.PolyData(vertices, faces_pv)

# 创建可视化器
plotter = pv.Plotter()

# 添加网格到可视化器，初始化时显示第一帧
frame_idx = 0
frame_color_mappings = color_mappings[frame_idx * vertices.shape[0]:(frame_idx + 1) * vertices.shape[0]]
frame_opacity_mappings = opacity_mappings[frame_idx * vertices.shape[0]:(frame_idx + 1) * vertices.shape[0]]

# 获取该帧的颜色和透明度
face_colors = np.mean(frame_color_mappings[faces], axis=1)
face_opacities = np.mean(frame_opacity_mappings[faces], axis=1)

# 创建 RGBA 数据（颜色和透明度合并）
rgba_colors = np.hstack([face_colors, face_opacities.reshape(-1, 1)])

# 将 RGBA 数据添加到网格的面数据中
mesh.cell_data['color'] = rgba_colors

# 添加网格到可视化器
plotter.add_mesh(mesh, scalars='color', rgb=True)

# 显示初始网格
plotter.show(auto_close=False)

# 循环渲染所有帧
for frame_idx in range(frame_count):
    print(f"渲染第 {frame_idx + 1} 帧...")

    # 每一帧的颜色和透明度映射
    frame_color_mappings = color_mappings[frame_idx * vertices.shape[0]:(frame_idx + 1) * vertices.shape[0]]
    frame_opacity_mappings = opacity_mappings[frame_idx * vertices.shape[0]:(frame_idx + 1) * vertices.shape[0]]

    # 获取该帧的颜色和透明度
    face_colors = np.mean(frame_color_mappings[faces], axis=1)
    face_opacities = np.mean(frame_opacity_mappings[faces], axis=1)

    # 创建 RGBA 数据（颜色和透明度合并）
    rgba_colors = np.hstack([face_colors, face_opacities.reshape(-1, 1)])

    # 更新网格的颜色和透明度
    mesh.cell_data['color'] = rgba_colors

    # 更新渲染
    plotter.update_scalars(mesh.cell_data['color'], render=False)
    plotter.render()

    # 等待 3 秒钟（每帧持续时间）
    time.sleep(frame_duration)

# 关闭可视化器
plotter.close()
print("渲染完成，关闭可视化器。")
