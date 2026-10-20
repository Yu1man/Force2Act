from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from sim_collect.task_specs import resolve_task_spec


def _decode_names(values) -> list[str]:
    output = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        output.append(str(value))
    return output


def _check_finite(name: str, array: np.ndarray) -> None:
    if not np.issubdtype(array.dtype, np.number):
        return
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


def validate_dataset(path: str | Path, task: str) -> dict[str, int]:
    spec = resolve_task_spec(task)
    path = Path(path).expanduser().resolve()

    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError("Missing /data group")
        if "mask" not in f:
            raise KeyError("Missing /mask group")
        if "train" not in f["mask"] or "valid" not in f["mask"]:
            raise KeyError("Force2Act training requires /mask/train and /mask/valid")

        train = _decode_names(f["mask"]["train"][()])
        valid = _decode_names(f["mask"]["valid"][()])
        if not train:
            raise ValueError("Training mask is empty")
        if not valid:
            raise ValueError("Validation mask is empty")
        if set(train) & set(valid):
            raise ValueError("Train and validation masks overlap")

        demos = {
            name
            for name in f["data"].keys()
            if name.startswith("demo_")
        }
        missing = (set(train) | set(valid)) - demos
        if missing:
            raise ValueError(f"Mask references missing demos: {sorted(missing)}")

        total_steps = 0
        for demo_name in sorted(set(train) | set(valid)):
            demo = f["data"][demo_name]
            if "actions" not in demo:
                raise KeyError(f"{demo_name}: missing actions")
            if "obs" not in demo:
                raise KeyError(f"{demo_name}: missing obs group")

            actions = np.asarray(demo["actions"])
            if actions.ndim != 2 or actions.shape[1] != spec.action_dim:
                raise ValueError(
                    f"{demo_name}: expected actions [T,{spec.action_dim}], got {actions.shape}"
                )
            _check_finite(f"{demo_name}/actions", actions)
            length = int(actions.shape[0])
            if length <= 0:
                raise ValueError(f"{demo_name}: empty trajectory")

            for key in (*spec.state_keys, *spec.force_keys):
                if key not in demo["obs"]:
                    raise KeyError(f"{demo_name}: missing obs/{key}")
                array = np.asarray(demo["obs"][key])
                if array.shape[0] != length:
                    raise ValueError(
                        f"{demo_name}/obs/{key}: T={array.shape[0]} != actions T={length}"
                    )
                _check_finite(f"{demo_name}/obs/{key}", array)

            for key in spec.camera_keys:
                if key not in demo["obs"]:
                    raise KeyError(f"{demo_name}: missing obs/{key}")
                image = np.asarray(demo["obs"][key])
                if image.shape[0] != length:
                    raise ValueError(
                        f"{demo_name}/obs/{key}: T={image.shape[0]} != actions T={length}"
                    )
                if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
                    raise ValueError(
                        f"{demo_name}/obs/{key}: expected [T,H,W,C], got {image.shape}"
                    )

            if "timestamps" in demo["obs"]:
                timestamps = np.asarray(demo["obs"]["timestamps"]).reshape(-1)
                if timestamps.shape[0] != length:
                    raise ValueError(
                        f"{demo_name}/obs/timestamps: T={timestamps.shape[0]} != {length}"
                    )

            total_steps += length

    return {
        "train_demos": len(train),
        "valid_demos": len(valid),
        "total_steps": total_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a collected HDF5 against the current Force2Act loader schema."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--hdf5", required=True)
    args = parser.parse_args()

    summary = validate_dataset(args.hdf5, args.task)
    print("Force2Act dataset validation passed")
    print(f"  train demos: {summary['train_demos']}")
    print(f"  valid demos: {summary['valid_demos']}")
    print(f"  transitions: {summary['total_steps']}")


if __name__ == "__main__":
    main()
