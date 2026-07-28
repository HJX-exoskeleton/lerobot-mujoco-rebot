"""Train or deploy Pi0 (from 7.pi0.ipynb)."""

import argparse

from scripts.cache_env import configure_project_caches

# Must run before importing LeRobot/Transformers/Hugging Face modules.
configure_project_caches()

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

from scripts.vla_runtime import add_subcommands, deploy, run_train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_subcommands(parser, "pi0_omy.yaml", "ckpt/pi0_omy/checkpoints/last/pretrained_model")
    args = parser.parse_args()
    if args.command == "train":
        run_train(args.config, args.pretrained, "pi0", args.cache_dir)
    else:
        deploy(args, PI0Policy, adapt_physical_gripper=True)


if __name__ == "__main__":
    main()
