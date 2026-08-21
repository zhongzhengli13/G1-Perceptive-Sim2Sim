# sim2sim

**中文** | [English](README_EN.md)

面向 Unitree G1 29DoF 的 MuJoCo Sim2Sim 验证示例。本仓库只负责加载已有 ONNX 策略并在 MuJoCo 中运行。

## 范围

包含：

- `stand` 和 `parkour` 两种运行入口。
- MuJoCo 场景、G1 29DoF 模型和 ONNX 文件放置目录。
- 键盘速度指令。
- 可替换的 actor 和 depth encoder ONNX 路径。

不包含：

- 训练代码。
- reward、domain randomization、训练日志或 checkpoint。
- 真机部署代码。

## 环境依赖

建议直接复用你已经用于训练和策略导出的 conda 环境。

```bash
conda activate <your_conda_env>
cd /path/to/sim2sim
```

如果你在训练工程根目录执行过下面的安装命令，`NumPy`、`OpenCV` 等项目依赖通常已经在当前环境中安装完成：

```bash
python -m pip install -e source/instinctlab
```

如果当前环境尚未安装 MuJoCo，执行：

```bash
python -m pip install mujoco
```

如果已经安装过 MuJoCo，可以跳过这一步。然后安装 ONNX Runtime：

```bash
python -m pip install onnxruntime
```

使用以下命令检查依赖是否安装成功：

```bash
python -c "import mujoco, onnxruntime; print('MuJoCo:', mujoco.__version__)"
```

## 快速开始

运行前请先按“准备 ONNX 策略”放置对应模型文件。

运行 parkour policy：

```bash
python3 sim2sim.py --task parkour
```

运行站立 policy：

```bash
python3 sim2sim.py --task stand
```

parkour 模式键盘控制：

| 按键 | 功能 |
| --- | --- |
| `↑` / `KP8` | 增加前向速度 |
| `↓` / `KP2` | 减小前向速度 |
| `←` / `KP4` | 增加左转角速度 |
| `→` / `KP6` | 增加右转角速度 |
| `Space` / `KP5` | 速度命令归零 |
| `R` | 重置 |

## 准备 ONNX 策略

本仓库不提供 ONNX 策略文件。请自行准备 actor 和 depth encoder，并放入 `policy/` 目录，或在 `config.py` 中修改路径。

parkour 模式默认读取：

```text
policy/parkour_actor.onnx
policy/parkour_depth_encoder.onnx
```

stand 模式默认读取：

```text
policy/stand_actor.onnx
policy/stand_depth_encoder.onnx
```

也可以保持自己的文件路径不变，在 `config.py` 中修改：

```python
PARKOUR_POLICY_FILE = "/path/to/exported/actor.onnx"
PARKOUR_DEPTH_ENCODER_FILE = "/path/to/exported/0-depth_encoder.onnx"
```

actor 和 depth encoder 应来自同一次训练或兼容配置。

## 目录结构

```text
.
|-- config.py
|-- mujoco_env.py
|-- onnx_inference.py
|-- sim2sim.py
|-- policy/
|-- g1_29dof/
|-- requirements.txt
|-- NOTICE.md
`-- LICENSE
```

## 常见问题

### 无法加载 ONNX

检查：

- `PARKOUR_POLICY_FILE` 是否指向实际存在的 actor ONNX。
- `PARKOUR_DEPTH_ENCODER_FILE` 是否指向实际存在的 depth encoder ONNX。
- 是否安装了 `onnxruntime`。
- actor 和 depth encoder 是否兼容。

## 开源边界

本仓库源码和文档使用 MIT License。你可以自由使用、复制、修改和分发本仓库代码，但需要保留原始版权声明和许可证文本。

请注意：

- 未经授权，请勿将本仓库中的策略权重、机器人模型、mesh 或其他第三方资产用于商业用途。
- ONNX 策略权重、`g1_29dof/*.xml` 和 `g1_29dof/meshes_recv_1_0/*.STL` 可能有各自的上游授权，公开再分发或商业使用前请自行确认许可边界。
- 本仓库不公开训练 reward、domain randomization、训练日志、checkpoint 或真机部署参数。
- 本仓库仅用于学习、研究和 Sim2Sim 验证，不构成真机部署建议。

## 共创方向

欢迎围绕以下方向共创： 

- 更清晰的安装和运行文档。
- 更多 MuJoCo 测试场景。
- 更稳定的示例策略接入方式。
- 不同系统和环境下的兼容性修复。
- 更轻量的 demo 资产组织方式。
