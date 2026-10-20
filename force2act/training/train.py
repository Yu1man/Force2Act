from force2act.training.utils.config import apply_cli_overrides, load_config, parse_args
from force2act.training.utils.runner import main


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    main(config, args)
