# sim2sim

[中文](README.md) | **English**

A MuJoCo Sim2Sim validation example for Unitree G1 29DoF. This repository only loads existing ONNX policies and runs them in MuJoCo.

## Scope

Included:

- `stand` and `parkour` entry points.
- MuJoCo scenes, the G1 29DoF model, and an ONNX file location convention.
- Keyboard velocity commands.
- Replaceable actor and depth encoder ONNX paths.

Not included:

- Training code.
- Rewards, domain randomization, training logs, or checkpoints.
- Real-robot deployment code.

## Environment

Reuse the conda environment that you already use for training and policy export.

```bash
conda activate <your_conda_env>
cd /path/to/sim2sim
```

If you have run the following command from the training project root, dependencies such as `NumPy` and `OpenCV` are usually already installed in the current environment:

```bash
python -m pip install -e source/instinctlab
```

If MuJoCo is not installed in the current environment, run:

```bash
python -m pip install mujoco
```

If MuJoCo is already installed, skip that step. Then install ONNX Runtime:

```bash
python -m pip install onnxruntime
```

Check the dependencies:

```bash
python -c "import mujoco, onnxruntime; print('MuJoCo:', mujoco.__version__)"
```

## Quick Start

Place the required ONNX model files first. See "Preparing ONNX Policies".

Run the parkour policy:

```bash
python3 sim2sim.py --task parkour
```

Run the stand policy:

```bash
python3 sim2sim.py --task stand
```

Keyboard controls in parkour mode:

| Key | Action |
| --- | --- |
| `Up` / `KP8` | Increase forward velocity |
| `Down` / `KP2` | Decrease forward velocity |
| `Left` / `KP4` | Increase left yaw command |
| `Right` / `KP6` | Increase right yaw command |
| `Space` / `KP5` | Stop |
| `R` | Reset |

## Preparing ONNX Policies

This repository does not provide ONNX policy files. Prepare the actor and depth encoder yourself, then place them under `policy/` or update the paths in `config.py`.

Parkour mode reads these files by default:

```text
policy/parkour_actor.onnx
policy/parkour_depth_encoder.onnx
```

Stand mode reads these files by default:

```text
policy/stand_actor.onnx
policy/stand_depth_encoder.onnx
```

You can also keep your own file paths and update `config.py`:

```python
PARKOUR_POLICY_FILE = "/path/to/exported/actor.onnx"
PARKOUR_DEPTH_ENCODER_FILE = "/path/to/exported/0-depth_encoder.onnx"
```

The actor and depth encoder should come from the same training run or compatible configurations.

## Repository Layout

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

## Troubleshooting

### ONNX Cannot Be Loaded

Check:

- `PARKOUR_POLICY_FILE` points to an existing actor ONNX file.
- `PARKOUR_DEPTH_ENCODER_FILE` points to an existing depth encoder ONNX file.
- `onnxruntime` is installed.
- Actor and depth encoder are compatible.

## Open Source Boundary

The source code and documentation are released under the MIT License. You may use, copy, modify, and distribute the code, provided that the original copyright notice and license text are retained.

Please note:

- Do not use policy weights, robot models, meshes, or other third-party assets in this repository for commercial purposes without proper authorization.
- ONNX policy weights, `g1_29dof/*.xml`, and `g1_29dof/meshes_recv_1_0/*.STL` may be subject to their own upstream licenses. Please verify redistribution and commercial-use permissions before publishing or using them commercially.
- This repository does not publish training rewards, domain randomization settings, training logs, checkpoints, or real-robot deployment parameters.
- This repository is intended for learning, research, and Sim2Sim validation only. It is not a real-robot deployment recommendation.

## Contribution Directions

Contributions are welcome around:

- Clearer setup and runtime documentation.
- Additional MuJoCo test scenes.
- More stable demo policy integration.
- Compatibility fixes across different systems and environments.
- Lighter demo asset organization.
