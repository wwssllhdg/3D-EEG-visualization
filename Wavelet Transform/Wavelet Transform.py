import pywt
import numpy as np

def compute_wavelet_power_single_trial(eeg_data, fs=200):  # 设置采样率为 200Hz
    """
    使用小波变换计算不同频段的功率（针对单个试验的数据）。
    eeg_data: 形状 (21, 104000) 的 EEG 信号（只包含一个试验的数据）。
    fs: 采样率（200Hz）。
    返回值: 每个电极的 δ, θ, α, β, γ 波段功率，形状 (21, 5, 104000)。
    """
    freqs = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 100)
    }

    wavelet = 'cmor1.5-1.0'  # 带宽 1.5，中心频率 1.0
    # Morlet小波是傅里叶变换和平滑窗结合的复小波，适用于EEG、语音信号分析等时频变化较强的信号。
    power_data = np.zeros((eeg_data.shape[0], 5, eeg_data.shape[1]))  # (21, 5, 104000)

    print(f"Starting wavelet transform for {eeg_data.shape[0]} electrodes, "
          f"each with {eeg_data.shape[1]} time points.")

    for electrode in range(eeg_data.shape[0]):  # 对每个电极进行处理
        print(f"  Processing electrode {electrode + 1}/{eeg_data.shape[0]}...")
        signal = eeg_data[electrode, :]

        # 小波变换
        scales = np.arange(2, 128)
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet, 1 / fs)
        # pywt.cwt
        # 进行小波变换，返回：
        # coefficients：小波变换的系数，表示信号在各个尺度上的表现。
        # frequencies：对应的频率值。

        # 使用enumerate遍历频段字典freqs，计算每个频段的功率
        for i, (band, (f_low, f_high)) in enumerate(freqs.items()):
            band_mask = (frequencies >= f_low) & (frequencies <= f_high)
            band_power = np.mean(np.abs(coefficients[band_mask, :]) ** 2, axis=0)
            power_data[electrode, i, :] = band_power


    return power_data

# 加载 EEG 数据（形状为 (21, 104000)）
eeg_data = np.load('../DataPorcess/new_sub01.npy')  # 加载经过平均参考处理后的 EEG 数据
print("\nLoaded EEG data shape:", eeg_data.shape)

# 计算功率
power_data = compute_wavelet_power_single_trial(eeg_data)

# 输出 eeg_data 和 power_data 的形状和部分数据
print("\nFinal eeg_data shape:", eeg_data.shape)  # 输出 eeg_data 的形状
print("Final power_data shape:", power_data.shape)  # 输出 power_data 的形状
# Final eeg_data shape: (21, 104000)
# Final power_data shape: (21, 5, 104000)

# 打印 eeg_data 和 power_data 的部分数据，查看内容
print("\n部分 eeg_data 内容：")
print(eeg_data[:, :5])  # 打印前 5 个时间点的 EEG 数据

print("\n部分 power_data 内容：")
print(power_data[:, 0, :5])  # 打印第一个电极、δ波（第 1 个频段）的前 5 个时间点的功率数据

np.save('power_data.npy', power_data)
print("\npower_data 已保存为 power_data.npy")