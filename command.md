# Force2Act Useful Commands

> Run these from the Force2Act repository root. Replace paths when needed.
> Model architecture (`no_force`, `cross_attn`, `force_adaln`, `fsra`) is set in the YAML config.

## List Simulation Collection Tasks

```bash
python -m sim_collect.collect --list
```

## Weight Collection Help

```bash
python -m sim_collect.collect weight-sort --help
```

## Door Collection Help

```bash
python -m sim_collect.collect door --help
```

## Nut Assembly Collection Help

```bash
python -m sim_collect.collect nut-assembly --help
```

## Collect Weight Dataset

```bash
python -m sim_collect.collect weight-sort \
    --output data/sim/weight_sort.hdf5 \
    --num-episodes 100 \
    --overwrite
```

## Collect Door Dataset

```bash
python -m sim_collect.collect door \
    --output data/sim/door_push_pull.hdf5 \
    --episodes 100 \
    --overwrite
```

## Collect Nut Assembly Dataset

```bash
python -m sim_collect.collect nut-assembly \
    --source /path/to/robomimic/nut_assembly_square.hdf5 \
    --output data/sim/nut_assembly.hdf5 \
    --num-demos 100 \
    --val-ratio 0.1 \
    --split-seed 42 \
    --overwrite
```

## Validate Weight Dataset

```bash
python -m sim_collect.validate \
    --task weight-sort \
    --hdf5 data/sim/weight_sort.hdf5
```

## Validate Door Dataset

```bash
python -m sim_collect.validate \
    --task door \
    --hdf5 data/sim/door_push_pull.hdf5
```

## Validate Nut Assembly Dataset

```bash
python -m sim_collect.validate \
    --task nut-assembly \
    --hdf5 data/sim/nut_assembly.hdf5
```

## Count Model Parameters

```bash
python -m tests.count_parameters
```

## Training Smoke Test

```bash
python -m force2act.training.train \
    --config configs/test.yaml \
    --max-steps 20
```

## Train Normally

```bash
python -m force2act.training.train \
    --config configs/test.yaml
```

## Train With Dataset Override

```bash
python -m force2act.training.train \
    --config configs/test.yaml \
    --hdf5-path data/sim/weight_sort.hdf5
```

## Train Seed 42

```bash
python -m force2act.training.train \
    --config configs/test.yaml \
    --seed 42 \
    --output-dir outputs/weight/fsra_seed42 \
    --wandb-name weight_fsra_seed42
```

## Train Seed 123

```bash
python -m force2act.training.train \
    --config configs/test.yaml \
    --seed 123 \
    --output-dir outputs/weight/fsra_seed123 \
    --wandb-name weight_fsra_seed123
```

## Train Seed 456

```bash
python -m force2act.training.train \
    --config configs/test.yaml \
    --seed 456 \
    --output-dir outputs/weight/fsra_seed456 \
    --wandb-name weight_fsra_seed456
```

## Train With Custom Batch Size and Learning Rate

```bash
python -m force2act.training.train \
    --config configs/test.yaml \
    --batch-size 64 \
    --lr 1e-4
```

## Train Without W&B

```bash
python -m force2act.training.train \
    --config configs/test.yaml \
    --no-wandb
```

## Train Explicitly on CUDA

```bash
python -m force2act.training.train \
    --config configs/test.yaml \
    --device cuda
```

## Resume Training

```bash
python -m force2act.training.train \
    --config configs/test.yaml \
    --resume outputs/weight/fsra_seed42/latest.pt
```

## Quick Rollout Check

```bash
python -m evaluation.sim.rollout \
    --checkpoint outputs/weight/fsra_seed42/final.pt \
    --exec-horizon 16 \
    --sampling-steps 20 \
    --episodes 5 \
    --seed 104729
```

## Rollout Check With Video

```bash
python -m evaluation.sim.rollout \
    --checkpoint outputs/weight/fsra_seed42/final.pt \
    --exec-horizon 16 \
    --sampling-steps 20 \
    --episodes 5 \
    --seed 104729 \
    --video outputs/weight/fsra_seed42/debug.mp4
```

## Publication Ablation

```bash
python -m evaluation.sim.ablation \
    --checkpoints \
        outputs/door/fsra_seed42/final.pt \
        outputs/door/fsra_seed123/final.pt \
        outputs/door/fsra_seed456/final.pt \
    --exec-horizons 4 16 \
    --sampling-steps 20 \
    --eval-seeds 104729 130363 155921 196613 262147 \
    --output-dir results/door/fsra
```

## Save All Validation Videos

```bash
python -m evaluation.sim.videos \
    --checkpoint outputs/door/fsra_seed42/final.pt \
    --exec-horizon 16 \
    --sampling-steps 20 \
    --output-dir videos/door/fsra_seed42
```