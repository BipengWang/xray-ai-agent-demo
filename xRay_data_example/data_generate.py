import numpy as np
import pandas as pd

# 随机生成能量轴 (1.0 到 10.0)
energy = np.linspace(1, 10, 100)

# 随机生成强度 + 一个高斯峰
peak_center = 5.0 + np.random.uniform(-0.5, 0.5)
peak_width = 0.5 + np.random.uniform(0.0, 0.3)
intensity = np.exp(-0.5 * ((energy - peak_center) / peak_width)**2)

# 加一点随机噪声
intensity += 0.05 * np.random.rand(len(energy))

df = pd.DataFrame({
    "energy": energy,
    "intensity": intensity
})

df.to_csv("spectrum.csv", index=False)
print("Generated spectrum.csv")
