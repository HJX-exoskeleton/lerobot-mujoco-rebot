"""Train or deploy SmolVLA (from 8.smolvla.ipynb)."""

import argparse

from scripts.cache_env import configure_project_caches

# Must run before importing LeRobot/Transformers/Hugging Face modules.
configure_project_caches()

from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from scripts.vla_runtime import add_subcommands, deploy, run_train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_subcommands(parser, "smolvla_omy.yaml", "ckpt/smolvla_omy/checkpoints/last/pretrained_model")
    args = parser.parse_args()
    if args.command == "train":
        run_train(args.config, args.pretrained, "smolvla", args.cache_dir)
    else:
        deploy(args, SmolVLAPolicy, adapt_physical_gripper=True)


if __name__ == "__main__":
    main()
