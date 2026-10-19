import argparse
import random
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from force2act.data.image_preprocessing import preprocess_images
from force2act.data.normalization import (
    ModalityStats,
    StandardNormalizer,
    fit_normalizer,
)
from force2act.data.split import get_train_val_names
from force2act.data.hdf5 import load_episodes
from force2act.data.dataset import EpisodeDataset

from force2act.models.policy.flow_matching import flow_matching_loss
from force2act.models.factory import build_model




def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def prepare_batch(
    batch,
    normalizer,
    device,
    use_force=True,
):
    images = batch["images"].to(
        device,
        non_blocking=True,
    )
    images = preprocess_images(images)

    state = batch["state"].to(
        device,
        non_blocking=True,
    )
    state = normalizer.normalize(
        state,
        "state",
    )

    actions = batch["actions"].to(
        device,
        non_blocking=True,
    )
    actions = normalizer.normalize(
        actions,
        "actions",
    )

    action_mask = batch["action_mask"].to(
        device,
        non_blocking=True,
    )

    if use_force:
        force = batch["force"].to(
            device,
            non_blocking=True,
        )
        force = normalizer.normalize(
            force,
            "force",
        )
    else:
        force = None

    return {
        "images": images,
        "state": state,
        "force": force,
        "actions": actions,
        "action_mask": action_mask,
    }




def train_step(
    model,
    batch,
    normalizer,
    optimizer,
    device,
    use_force,
    grad_clip_norm=None,
):
    model.train()

    processed_batch = prepare_batch(
        batch=batch,
        normalizer=normalizer,
        device=device,
        use_force=use_force,
    )

    optimizer.zero_grad(set_to_none=True)

    loss = flow_matching_loss(
        model=model,
        actions=processed_batch["actions"],
        images=processed_batch["images"],
        state=processed_batch["state"],
        force=processed_batch["force"],
        action_mask=processed_batch["action_mask"],
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite training loss: {loss.item()}"
        )

    loss.backward()

    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip_norm,
        )

    optimizer.step()

    return loss.item()



def load_train_val_episodes(config):
    dataset_cfg = config["dataset"]

    hdf5_path = dataset_cfg["hdf5_path"]
    state_keys = dataset_cfg["state_keys"]
    camera_keys = dataset_cfg["camera_keys"]
    force_keys = dataset_cfg.get("force_keys")

    train_names, val_names = get_train_val_names(
        hdf5_path
    )

    train_episodes = load_episodes(
        hdf5_path=hdf5_path,
        demo_names=train_names,
        state_keys=state_keys,
        camera_keys=camera_keys,
        force_keys=force_keys,
    )

    val_episodes = load_episodes(
        hdf5_path=hdf5_path,
        demo_names=val_names,
        state_keys=state_keys,
        camera_keys=camera_keys,
        force_keys=force_keys,
    )

    return train_episodes, val_episodes


def build_dataloaders(
    config,
    train_dataset,
    val_dataset,
):
    training_cfg = config["training"]

    batch_size = int(
        training_cfg["batch_size"]
    )

    num_workers = int(
        training_cfg.get(
            "num_workers",
            0,
        )
    )

    seed = int(
        training_cfg.get(
            "seed",
            42,
        )
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common_args,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_args,
    )

    if len(train_loader) == 0:
        raise RuntimeError(
            "Training DataLoader is empty. "
            "Reduce batch_size."
        )

    if len(val_loader) == 0:
        raise RuntimeError(
            "Validation DataLoader is empty."
        )

    return train_loader, val_loader


def infinite_loader(loader):
    while True:
        yield from loader



def validate(
    model,
    val_loader,
    normalizer,
    device,
    use_force,
    max_batches=None,
):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    num_batches = 0

    if max_batches is None:
        iterator = val_loader
        total = len(val_loader)
    else:
        max_batches = min(
            int(max_batches),
            len(val_loader),
        )

        iterator = islice(
            val_loader,
            max_batches,
        )

        total = max_batches

    try:
        with torch.no_grad():
            for batch in tqdm(
                iterator,
                total=total,
                desc="Validation",
                leave=False,
                dynamic_ncols=True,
            ):
                processed_batch = prepare_batch(
                    batch=batch,
                    normalizer=normalizer,
                    device=device,
                    use_force=use_force,
                )

                loss = flow_matching_loss(
                    model=model,
                    actions=processed_batch["actions"],
                    images=processed_batch["images"],
                    state=processed_batch["state"],
                    force=processed_batch["force"],
                    action_mask=processed_batch[
                        "action_mask"
                    ],
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite validation loss."
                    )

                total_loss += loss.item()
                num_batches += 1

    finally:
        model.train(was_training)

    if num_batches == 0:
        raise RuntimeError(
            "Validation processed zero batches."
        )

    return total_loss / num_batches



def normalizer_state_dict(normalizer):
    state = {}

    for name, stats in normalizer.stats.items():
        state[name] = {
            "mean": stats.mean.detach().cpu(),
            "std": stats.std.detach().cpu(),
        }

    return state


def normalizer_from_state_dict(state):
    stats = {}

    for name, values in state.items():
        stats[name] = ModalityStats(
            mean=values["mean"],
            std=values["std"],
        )

    return StandardNormalizer(stats)




def save_checkpoint(
    path,
    model,
    optimizer,
    normalizer,
    config,
    step,
    best_val_loss,
    wandb_run_id=None,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalizer": normalizer_state_dict(
                normalizer
            ),
            "config": config,
            "step": int(step),
            "best_val_loss": float(
                best_val_loss
            ),
            "wandb_run_id": wandb_run_id,
        },
        path,
    )



def init_wandb(
    config,
    output_dir,
    resume_run_id=None,
):
    wandb_cfg = config.get(
        "wandb",
        {},
    )

    if not wandb_cfg.get(
        "enabled",
        True,
    ):
        return None

    try:
        import wandb

    except ImportError as exc:
        raise ImportError(
            "W&B is enabled but wandb is not installed.\n"
            "Run: python -m pip install wandb"
        ) from exc

    kwargs = {
        "project": wandb_cfg.get(
            "project",
            "Force2Act",
        ),
        "name": wandb_cfg.get("name"),
        "config": config,
        "dir": str(output_dir),
    }

    entity = wandb_cfg.get("entity")

    if entity:
        kwargs["entity"] = entity

    if resume_run_id is not None:
        kwargs["id"] = resume_run_id
        kwargs["resume"] = "allow"

    return wandb.init(**kwargs)



def load_config(config_path):
    with open(
        config_path,
        "r",
    ) as f:
        return yaml.safe_load(f)


def save_config(
    config,
    path,
):
    with open(
        path,
        "w",
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--hdf5-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
    )

    return parser.parse_args()


def apply_cli_overrides(
    config,
    args,
):
    training_cfg = config.setdefault(
        "training",
        {},
    )

    dataset_cfg = config.setdefault(
        "dataset",
        {},
    )

    wandb_cfg = config.setdefault(
        "wandb",
        {},
    )

    if args.max_steps is not None:
        training_cfg["max_steps"] = (
            args.max_steps
        )

    if args.batch_size is not None:
        training_cfg["batch_size"] = (
            args.batch_size
        )

    if args.lr is not None:
        training_cfg["learning_rate"] = (
            args.lr
        )

    if args.seed is not None:
        training_cfg["seed"] = args.seed

    if args.output_dir is not None:
        training_cfg["output_dir"] = (
            args.output_dir
        )

    if args.hdf5_path is not None:
        dataset_cfg["hdf5_path"] = (
            args.hdf5_path
        )

    if args.wandb_name is not None:
        wandb_cfg["name"] = (
            args.wandb_name
        )

    if args.no_wandb:
        wandb_cfg["enabled"] = False

    return config




def main(
    config,
    args,
):
    training_cfg = config["training"]
    dataset_cfg = config["dataset"]

    validation_cfg = config.get(
        "validation",
        {},
    )

    seed = int(
        training_cfg.get(
            "seed",
            42,
        )
    )

    seed_everything(seed)



    if args.device is not None:
        device = torch.device(
            args.device
        )
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )



    output_dir = Path(
        training_cfg.get(
            "output_dir",
            "outputs/default",
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_config(
        config,
        output_dir / "config.yaml",
    )



    checkpoint = None

    if args.resume is not None:
        print(
            f"Loading checkpoint: {args.resume}"
        )

        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

    print("Loading episodes...")

    train_episodes, val_episodes = (
        load_train_val_episodes(config)
    )

    print(
        f"Train episodes: "
        f"{len(train_episodes)}"
    )

    print(
        f"Val episodes: "
        f"{len(val_episodes)}"
    )



    if checkpoint is not None:
        normalizer = (
            normalizer_from_state_dict(
                checkpoint["normalizer"]
            )
        )
    else:
        normalizer = fit_normalizer(
            train_episodes
        )

    normalizer.to(device)



    history = int(
        dataset_cfg["history"]
    )

    action_horizon = int(
        dataset_cfg["action_horizon"]
    )

    train_dataset = EpisodeDataset(
        train_episodes,
        history,
        action_horizon,
    )

    val_dataset = EpisodeDataset(
        val_episodes,
        history,
        action_horizon,
    )

    print(
        f"Train windows: "
        f"{len(train_dataset)}"
    )

    print(
        f"Val windows: "
        f"{len(val_dataset)}"
    )

    train_loader, val_loader = (
        build_dataloaders(
            config,
            train_dataset,
            val_dataset,
        )
    )

    train_iterator = infinite_loader(
        train_loader
    )


    model = build_model(
        config
    ).to(device)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "Model has no trainable parameters."
        )

    num_trainable = sum(
        parameter.numel()
        for parameter
        in trainable_parameters
    )

    print(
        f"Trainable parameters: "
        f"{num_trainable:,}"
    )


    learning_rate = float(
        training_cfg[
            "learning_rate"
        ]
    )

    weight_decay = float(
        training_cfg.get(
            "weight_decay",
            1e-4,
        )
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )



    max_steps = int(
        training_cfg["max_steps"]
    )

    log_interval = int(
        training_cfg.get(
            "log_interval",
            20,
        )
    )

    val_interval = int(
        training_cfg.get(
            "val_interval",
            1000,
        )
    )

    checkpoint_interval = int(
        training_cfg.get(
            "checkpoint_interval",
            1000,
        )
    )

    grad_clip_norm = (
        training_cfg.get(
            "grad_clip_norm"
        )
    )

    if grad_clip_norm is not None:
        grad_clip_norm = float(
            grad_clip_norm
        )

    max_val_batches = (
        validation_cfg.get(
            "max_batches"
        )
    )

    use_force = bool(
        dataset_cfg.get(
            "use_force",
            True,
        )
    )

    if log_interval <= 0:
        raise ValueError(
            "log_interval must be > 0"
        )

    if val_interval <= 0:
        raise ValueError(
            "val_interval must be > 0"
        )

    if checkpoint_interval <= 0:
        raise ValueError(
            "checkpoint_interval must be > 0"
        )

    start_step = 0
    best_val_loss = float("inf")

    if checkpoint is not None:
        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        start_step = int(
            checkpoint.get(
                "step",
                0,
            )
        )

        best_val_loss = float(
            checkpoint.get(
                "best_val_loss",
                float("inf"),
            )
        )

        print(
            f"Resuming from step "
            f"{start_step}"
        )

    if start_step >= max_steps:
        print(
            "Checkpoint has already reached "
            "max_steps."
        )
        return


    resume_run_id = None

    if checkpoint is not None:
        resume_run_id = (
            checkpoint.get(
                "wandb_run_id"
            )
        )

    run = init_wandb(
        config=config,
        output_dir=output_dir,
        resume_run_id=resume_run_id,
    )

    wandb_run_id = (
        run.id
        if run is not None
        else None
    )


    print(
        f"Training from step "
        f"{start_step} → {max_steps}"
    )

    print(
        f"Validation every "
        f"{val_interval} steps"
    )

    running_loss = 0.0
    running_count = 0

    last_completed_step = start_step

    progress = tqdm(
        total=max_steps,
        initial=start_step,
        desc="Training",
        dynamic_ncols=True,
    )

    try:
        for step in range(
            start_step + 1,
            max_steps + 1,
        ):
            batch = next(
                train_iterator
            )

            train_loss = train_step(
                model=model,
                batch=batch,
                normalizer=normalizer,
                optimizer=optimizer,
                device=device,
                use_force=use_force,
                grad_clip_norm=grad_clip_norm,
            )

            last_completed_step = step

            running_loss += train_loss
            running_count += 1

            metrics = {}


            if (
                step % log_interval == 0
                or step == max_steps
            ):
                mean_train_loss = (
                    running_loss
                    / running_count
                )

                metrics[
                    "train/loss"
                ] = mean_train_loss

                metrics[
                    "train/learning_rate"
                ] = optimizer.param_groups[
                    0
                ]["lr"]

                running_loss = 0.0
                running_count = 0


            validation_due = (
                step % val_interval == 0
                or step == max_steps
            )

            if validation_due:
                val_loss = validate(
                    model=model,
                    val_loader=val_loader,
                    normalizer=normalizer,
                    device=device,
                    use_force=use_force,
                    max_batches=max_val_batches,
                )

                metrics[
                    "val/loss"
                ] = val_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    save_checkpoint(
                        path=(
                            output_dir
                            / "best.pt"
                        ),
                        model=model,
                        optimizer=optimizer,
                        normalizer=normalizer,
                        config=config,
                        step=step,
                        best_val_loss=(
                            best_val_loss
                        ),
                        wandb_run_id=(
                            wandb_run_id
                        ),
                    )

                    print(
                        f"\nNew best validation "
                        f"loss: "
                        f"{best_val_loss:.6f}"
                    )


            if (
                step
                % checkpoint_interval
                == 0
            ):
                save_checkpoint(
                    path=(
                        output_dir
                        / "latest.pt"
                    ),
                    model=model,
                    optimizer=optimizer,
                    normalizer=normalizer,
                    config=config,
                    step=step,
                    best_val_loss=(
                        best_val_loss
                    ),
                    wandb_run_id=(
                        wandb_run_id
                    ),
                )



            if (
                run is not None
                and metrics
            ):
                run.log(
                    metrics,
                    step=step,
                )

            progress.set_postfix(
                train=f"{train_loss:.4f}",
                best_val=(
                    "N/A"
                    if best_val_loss
                    == float("inf")
                    else (
                        f"{best_val_loss:.4f}"
                    )
                ),
            )

            progress.update(1)

    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. "
            "Saving latest checkpoint..."
        )

        save_checkpoint(
            path=(output_dir / "latest.pt"),
            model=model,
            optimizer=optimizer,
            normalizer=normalizer,
            config=config,
            step=last_completed_step,
            best_val_loss=best_val_loss,
            wandb_run_id=wandb_run_id,
        )

        return

    finally:
        progress.close()

        if run is not None:
            run.finish()


    save_checkpoint(
        path=(output_dir / "final.pt"),
        model=model,
        optimizer=optimizer,
        normalizer=normalizer,
        config=config,
        step=max_steps,
        best_val_loss=best_val_loss,
        wandb_run_id=wandb_run_id,
    )

    save_checkpoint(
        path=(output_dir/ "latest.pt"),
        model=model,
        optimizer=optimizer,
        normalizer=normalizer,
        config=config,
        step=max_steps,
        best_val_loss=best_val_loss,
        wandb_run_id=wandb_run_id,
    )

    print("Training finished.")
    print(f"Best validation loss: ", f"{best_val_loss:.6f}")




if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)

    config = apply_cli_overrides(config,args)

    main(config, args)