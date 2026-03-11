# Keystroke Vibration Data Collector

利用 Apple Silicon MacBook 内置的 BMI286 IMU（加速度计+陀螺仪）采集键盘敲击振动数据，
用于后续训练按键识别 AI 模型。

## 环境要求

- **硬件**: Apple Silicon MacBook (M1/M2/M3/M4)
- **系统**: macOS 14+
- **Python**: 3.10+
- **权限**: 需要 sudo（IOKit HID 设备访问）

## 安装

```bash
# 克隆本项目
cd keystroke_collector

# 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 同时确保 macimu 的上游仓库已安装
# pip install git+https://github.com/olvvier/apple-silicon-accelerometer.git
```

### 验证传感器

```bash
ioreg -l -w0 | grep -A5 AppleSPUHIDDevice
```

如果有输出，说明你的 Mac 支持此传感器。

## 使用方法

### 1. 单键重复模式（推荐先做）

程序会依次提示你按每个键 50 次，覆盖 a-z 和 0-9 共 36 个键。

```bash
sudo .venv/bin/python3 collector.py --mode single_key --participant p01
```

可选参数：
```bash
# 修改每个键的重复次数
sudo .venv/bin/python3 collector.py --mode single_key --repeats 30

# 修改最低采样率阈值（默认96Hz）
sudo .venv/bin/python3 collector.py --mode single_key --min-rate 90
```

### 2. 自由打字模式

自然打字，所有按键自动记录。按 Ctrl+C 结束。

```bash
sudo .venv/bin/python3 collector.py --mode free_type --participant p01
```

### 3. 数据预处理

采集完成后，运行预处理器将原始数据切窗对齐生成训练数据集：

```bash
python3 preprocessor.py --session data/raw/p01_single_key_20260306_143022
```

会生成：
- `data/processed/xxx_dataset.npz` — NumPy 压缩格式，可直接用于 PyTorch/TF
- `data/processed/xxx_flat.csv` — 扁平 CSV 格式，可直接用于 sklearn

## 采样率监控

- 实时监控 IMU 采样率（约 100Hz）
- 如果连续检测到采样率低于 96Hz，**自动停止录制并报警**
- 已录制的数据会被安全保存
- 会话元数据文件中记录了采样率的最小值/最大值/平均值

## 输出文件说明

每次采集会在 `data/raw/` 下生成三个文件：

| 文件 | 内容 |
|------|------|
| `*_sensor.csv` | 连续传感器数据流（timestamp_ns, accel_xyz, gyro_xyz） |
| `*_events.csv` | 键盘事件（timestamp_ns, key, press/release） |
| `*_meta.txt` | 会话元数据（时长、采样率统计、是否有效等） |

## 项目结构

```
keystroke_collector/
├── config.py            # 所有可调参数
├── sensor_reader.py     # IMU 传感器读取（加速度计+陀螺仪）
├── keyboard_listener.py # 键盘事件监听
├── rate_monitor.py      # 采样率监控 & 报警
├── collector.py         # 主采集器（入口）
├── preprocessor.py      # 后处理：切窗对齐、生成训练集
├── requirements.txt     # 依赖
└── data/
    ├── raw/             # 原始采集数据
    └── processed/       # 预处理后的训练数据
```

## 采集建议

1. **环境**: 找一个安静平稳的桌面，MacBook 平放
2. **姿势**: 正常打字姿势，双手放在键盘上
3. **单键模式**: 变化手指和力度（轻按、重按混合）
4. **自由打字**: 至少打 5-10 分钟，覆盖尽可能多的字母组合
5. **多次采集**: 建议不同时间段采集 2-3 次，增加数据多样性

## 数据格式（预处理后）

### NPZ 格式
```python
import numpy as np
data = np.load("data/processed/xxx_dataset.npz", allow_pickle=True)
X = data["X"]           # shape: (N, 30, 6) — N个样本 × 30个时间步 × 6通道
y = data["y"]           # shape: (N,)       — 按键标签
channels = data["channels"]  # ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z']
```

### 6 个传感器通道
| 通道 | 说明 | 单位 |
|------|------|------|
| accel_x | X轴加速度 | g |
| accel_y | Y轴加速度 | g |
| accel_z | Z轴加速度 | g |
| gyro_x | X轴角速度 | deg/s |
| gyro_y | Y轴角速度 | deg/s |
| gyro_z | Z轴角速度 | deg/s |
