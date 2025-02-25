import numpy as np

# 1) 载入原始 EEG 数据 (形状假设为 (7, 62, 104000))
all_data = np.load('sub1.npy')  # 您的原始数据文件

# 2) 定义所有通道的名称，顺序与 all_data 的第 1 维 (axis=1) 一致
channel_names = [
    "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7（10-20中的T3）",
    "C5","C3","C1","CZ","C2","C4","C6","T8（10-20中的T4）","TP7","CP5","CP3",
    "CP1","CPZ","CP2","CP4","CP6","TP8","P7（10-20中的T5）","P5","P3","P1",
    "PZ","P2","P4","P6","P8（10-20中的T6）","PO7","PO5","PO3","POZ","PO4",
    "PO6","PO8","CB1","O1","OZ","O2","CB2"
]

# 3) 定义您想提取的“标红”通道列表
red_channels = [
    "FP1", "FPZ", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T7（10-20中的T3）", "C3", "CZ", "C4", "T8（10-20中的T4）",
    "P7（10-20中的T5）", "P3", "PZ", "P4", "P8（10-20中的T6）",
    "O1", "OZ", "O2"
]

# 4) 找到 red_channels 在 channel_names 中对应的索引
indices = [channel_names.index(ch) for ch in red_channels]

# 5) 从 all_data 中提取这些通道 (沿 axis=1 选择通道)
new_data = all_data[:, indices, :]

# 6) 计算每个时间点的平均电压，axis=1 表示在通道维度上计算
average_voltage_per_time = np.mean(new_data, axis=1, keepdims=True)

# 7) 进行平均参考
average_referenced_data = new_data - average_voltage_per_time

# 提取第一个试验的数据 (假设新数据格式为 (7, 21, 104000))
first_trial_data = average_referenced_data[0, :, :]

# 8) 只取前 12000 个数据点
first_trial_data = first_trial_data[:, :200]

# 9) 保存为新的 .npy 文件
np.save('new_sub01.npy', first_trial_data)

print("处理完成，新数据已保存为 new_sub01.npy")
print("新数据格式为", first_trial_data.shape)
