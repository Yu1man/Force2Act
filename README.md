# Force2Act

**Force-Conditioned Flow Matching for Contact-Aware Robot Manipulation**

Force2Act is a research codebase for learning robot action trajectories from visual, proprioceptive, and force observations using conditional flow matching. The project is designed to support controlled comparisons between force-conditioned and no-force policies, with a clean separation between data handling, model architecture, flow-matching objectives, and robot-specific interfaces.

This repository is currently under active development. The present version contains the core data abstractions, preprocessing utilities, model components, a complete cross-attention Force2Act model, flow-matching training and sampling primitives, a configuration-driven model factory, and an HDF5-to-episode loader. Full training, validation, experiment logging, and real-robot ROS integration are being added next.

## 1. Research Motivation

Vision and proprioception provide strong information about robot state and scene geometry, but many manipulation tasks also depend on physical interaction that cannot be inferred reliably from appearance alone. Contact force can reveal properties such as object resistance, interaction state, and whether the robot is making appropriate contact with the environment.

Force2Act studies how force observations can condition generative robot action policies while keeping the rest of the policy architecture and training setup controlled. The current baseline uses a transformer-style action decoder conditioned on visual, proprioceptive, and optional force tokens, trained with a rectified-flow / flow-matching objective.

The repository is intentionally organized so that the same learning pipeline can later consume both simulation demonstrations and real Franka demonstrations converted from ROS recordings.

## 2. Current Architecture

The current implemented model is `CrossAttentionForce2Act`.

At each policy step, the model receives:

- multi-camera image history,
- robot state history,
- optional force history,
- a noisy action trajectory `x_t`,
- continuous flow time `t`.

The model predicts the velocity field used by the flow-matching objective.

Conceptually:

```text
Images ────────> VisionEncoder ─────┐
                                    │
State ─────────> StateEncoder ──────┼──> Context tokens
                                    │
Force ─────────> ForceEncoder ──────┘      (optional)

Noisy action x_t ─> ActionEncoder + action positional embedding
                                      │
Flow time t ───────> Fourier embedding│
                                      v
                         Cross-attention DiT blocks
                                      │
                                      v
                              Final velocity head
                                      │
                                      v
                              predicted dx/dt
```

### 2.1 Observation encoders

The current implementation includes:

- **VisionEncoder**: frozen `timm` image backbone (currently ResNet-18 by default), projected to the transformer hidden dimension and augmented with learned camera, temporal, and spatial positional embeddings.
- **StateEncoder**: MLP over low-dimensional robot state followed by learned temporal positional embeddings.
- **ForceEncoder**: temporal 1D convolutional encoder that preserves the force-history length and maps force measurements into the shared hidden dimension.
- **ActionEncoder**: MLP for the noisy action trajectory, followed by learned action-horizon positional embeddings.

### 2.2 Transformer blocks

The cross-attention model uses DiT-style blocks containing:

1. action-to-context cross-attention,
2. action self-attention,
3. feed-forward processing,
4. adaptive LayerNorm modulation conditioned on continuous flow time.

The adaptive modulation and final output projection use zero initialization. This makes each DiT block initially close to the identity mapping and the initial predicted velocity field close to zero.

### 2.3 Force and no-force variants

The same model class supports both force-conditioned and no-force experiments.

When `use_force: true`, force tokens are encoded and appended to the observation context. When `use_force: false`, no force encoder is instantiated and force input is rejected by the model interface. This keeps the no-force baseline explicit rather than silently discarding force tensors.

## 3. Flow Matching

Let:

- `x_1` be the normalized ground-truth action trajectory,
- `x_0 ~ N(0, I)` be Gaussian noise,
- `t ~ Uniform(0, 1)`.

The interpolation path is

```text
x_t = (1 - t) x_0 + t x_1
```

with target velocity

```text
v* = x_1 - x_0.
```

The network predicts

```text
v_theta(x_t, t | observations)
```

and is trained with mean squared error over valid action timesteps only. The action mask prevents right-padded future actions near episode boundaries from contributing to the loss.

At inference time, action trajectories are generated by starting from Gaussian noise and integrating the learned velocity field from `t = 0` to `t = 1`. The current implementation uses Euler integration.

## 4. Canonical Data Representation

Robot-specific storage formats are converted into a common `Episode` representation before being passed to the learning code.

```text
images:      [T, N_cam, C, H, W]
state:       [T, D_state]
force:       [T, D_force]
actions:     [T, D_action]
timestamps:  [T]
```

For a no-force episode, `force` is represented as `[T, 0]` rather than mixing model-specific logic into the data layer.

A training window produced from an episode has the form:

```text
images:       [N_cam, T_image, C, H, W]
state:        [T_state, D_state]
force:        [T_force, D_force]
actions:      [T_action, D_action]
action_mask:  [T_action]
```

The dataset repeats the earliest observation when a history window extends before the start of an episode and masks action padding when the requested action horizon extends beyond the end of an episode.

## 5. Current Dataset Support

The repository currently includes an HDF5 loader for Robomimic-style datasets.

A typical demonstration contains:

```text
data/
  demo_0/
    actions
    obs/
      robot0_eef_pos
      robot0_eef_quat
      robot0_gripper_qpos
      robot0_eef_force
      agentview_image
      robot0_eye_in_hand_image
```

Multiple state keys are concatenated along the feature dimension. Multiple camera streams are stacked along a new camera dimension.

For example:

```text
EEF position       [T, 3]
EEF quaternion     [T, 4]
Gripper position   [T, 2]
                   -------
State              [T, 9]
```

while two cameras become

```text
camera 0           [T, C, H, W]
camera 1           [T, C, H, W]
                   ----------------
images             [T, 2, C, H, W]
```

If a simulation dataset does not contain timestamps, sequential step indices are used as a fallback. Real-robot recordings will retain real timestamps during ROS bag conversion.

## 6. Configuration

Force2Act keeps the configuration pattern used in the original research code. Configurations are grouped into `dataset`, `model`, `training`, `evaluation`, and `wandb` sections.

Example:

```yaml
dataset:
  hdf5_path: datasets/robosuite/door_push_pull.hdf5

  state_keys:
    - robot0_eef_pos
    - robot0_eef_quat
    - robot0_gripper_qpos

  camera_keys:
    - agentview_image
    - robot0_eye_in_hand_image

  force_keys:
    - robot0_eef_force

  history: 3
  action_horizon: 16
  action_dim: 7
  use_force: true

  image_size: 224
  model_image_size: 224

model:
  # If omitted, the model factory defaults to cross_attn.
  architecture: cross_attn

  state_dim: 9
  force_dim: 3

  hidden_dim: 256
  num_layers: 8
  num_heads: 8

  backbone_name: resnet18
  pretrained: true
  freeze_backbone: true

  fourier_dim: 128
  frequency_scale: 1.0
  dropout: 0.0
  mlp_ratio: 4

  max_feature_height: 16
  max_feature_width: 16

training:
  seed: 42
  batch_size: 64
  total_steps: 25000
  learning_rate: 0.0001
  weight_decay: 0.0001
  grad_clip: 1.0
  amp: true

evaluation:
  sampling_steps: 20
  exec_horizon: 4

wandb:
  enabled: true
  project: Force2Act
```

### Independent observation histories

The original configuration uses a shared value:

```yaml
history: 3
```

The current model factory also supports optional modality-specific overrides:

```yaml
image_history: 3
state_history: 3
force_history: 5
```

If an override is omitted, it falls back to `history`. This keeps older experiment configurations compatible while allowing force-history ablations.

## 7. Repository Structure

The current code is organized approximately as follows:

```text
Force2Act/
├── configs/
│   └── ...
│
├── force2act/
│   ├── data/
│   │   ├── episode.py
│   │   ├── dataset.py
│   │   ├── hdf5.py
│   │   ├── normalization.py
│   │   └── preprocessing.py
│   │
│   └── models/
│       ├── cross_attention.py
│       ├── factory.py
│       ├── flow_matching.py
│       └── components/
│           ├── blocks.py
│           ├── context.py
│           ├── embeddings.py
│           └── encoders.py
│
├── requirements.txt
└── README.md
```

Additional training, evaluation, ROS conversion, and real-robot deployment modules will be added as the implementation progresses.

## 8. Installation

A Python 3.10+ environment is recommended.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for the CUDA version on the machine before or together with the remaining dependencies.

## 9. Running the Current Smoke Tests

The repository currently uses module-level smoke tests while the full test suite is being assembled.

### Cross-attention model

```bash
python -m force2act.models.cross_attention
```

The current smoke test checks:

- force-conditioned forward pass,
- no-force forward pass,
- zero output at initialization,
- strict no-force input validation,
- flow-matching loss,
- backward propagation,
- two optimizer steps to verify upstream gradient flow,
- Euler action sampling.

Expected model output shape for the current default setup:

```text
[B, 16, 7]
```

### Model factory

```bash
python -m force2act.models.factory
```

The factory reads the YAML configuration and constructs the requested model. If `model.architecture` is absent, the current default is `cross_attn`.

### HDF5 loader

```bash
python -m force2act.data.hdf5
```

For the current door dataset, a loaded demonstration should follow the canonical shapes, for example:

```text
images      [T, 2, 3, 224, 224]
state       [T, 9]
force       [T, 3]
actions     [T, 7]
timestamps  [T]
```

## 10. Normalization and Image Preprocessing

Low-dimensional state, force, and action statistics are fitted from the training episodes only. Validation and test data must never be used to fit normalization statistics.

Images are kept as raw `uint8` tensors in the canonical episode representation. Image preprocessing is applied only when preparing a model batch:

```text
uint8 image
    ↓
float conversion / 255
    ↓
ImageNet normalization
    ↓
VisionEncoder
```

The current default ImageNet normalization constants are:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

## 11. Planned Training Pipeline

The next implementation stage connects the completed components into a reproducible experiment pipeline:

```text
HDF5
  ↓
Episode loading
  ↓
train / validation split
  ↓
EpisodeDataset + DataLoader
  ↓
training-only normalizer fitting
  ↓
image preprocessing + numeric normalization
  ↓
model factory
  ↓
flow-matching loss
  ↓
optimizer / scheduler / AMP / gradient clipping
  ↓
validation
  ↓
checkpointing
  ↓
Weights & Biases logging
```

The intended experiment infrastructure will use fixed seeds, explicit train/validation/test separation, reproducible evaluation noise, and saved configuration and normalization statistics in each checkpoint.

## 12. Planned Real-Robot Integration

The learning code is intentionally independent of ROS.

Real Franka demonstrations will be collected through ROS 2 and stored first as raw `rosbag2` recordings containing the relevant streams, such as:

```text
robot command
robot state
force / wrench
camera streams
Quest teleoperation state
TF transforms
```

A separate conversion stage will synchronize these timestamped streams and create the same canonical `Episode` / HDF5 representation used by the simulation pipeline:

```text
Quest + Franka + cameras
        ↓
      ROS 2
        ↓
     rosbag2
        ↓
 timestamp synchronization
        ↓
 canonical Episode
        ↓
       HDF5
        ↓
   Force2Act training
```

This separation allows simulation and real-robot data to share the same model and training code while keeping hardware-specific assumptions outside the core ML package.

## 13. Experimental Status

The current repository should be treated as an implementation-in-progress rather than a released benchmark. Quantitative simulation and real-robot results are intentionally not included in this README yet. They will be added after the new training pipeline, controlled ablations, and real-robot evaluation are completed.

Planned comparisons include:

- no-force baseline,
- simple force-conditioned cross-attention,
- dedicated force-conditioning architectures,
- force-history ablations,
- force corruption / misalignment controls,
- simulation and real-robot evaluation.

## 14. Reproducibility Goals

The publication version of the repository is being developed with the following requirements:

- explicit random seeds,
- multiple independent experimental seeds,
- training-only normalization statistics,
- fixed validation protocol,
- configuration saved with checkpoints,
- parameter-count reporting,
- force/no-force controls with matched evaluation conditions,
- deterministic or fixed-randomness validation where practical,
- repeated real-robot trials with uncertainty reporting.

## Citation

A paper citation will be added once the manuscript is publicly available.

---

**Status:** active research development.