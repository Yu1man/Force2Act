# Modality

Observation history = 3

images:
[B, N_cam, T_obs, C, H, W]

state:
[B, T_obs, state_dim]

force:
[B, T_obs, force_dim]

# Target
actions:
[B, T_action, 7]

action = [dx, dy, dz, rx, ry, rz, gripper]

action horizon = 16

# Exectuion
- Action Horion = 16
- Exec Horion = 4
- Control / replanning frequency = TBD after Franka interface test

# Action semantics
- Translation frame = TBD
- Rotation representation = TBD
- Translation units = TBD
- Gripper convention = TBD