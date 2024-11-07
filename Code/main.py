import argparse
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from hierarchical_env import HierarchicalTNNTNEnv


def create_env(env_name):
    if env_name == "hierarchical":
        return HierarchicalTNNTNEnv()
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def main(args):
    ray.init(_temp_dir=os.path.abspath("./tmp"), include_dashboard=False)

    env = create_env(args.env)
    register_env("tn_ntn_env", lambda config: create_env(args.env))

    config = (
        PPOConfig()
        .environment("tn_ntn_env")
        .framework("torch")
        .multi_agent(
            policies={
                "satellite": (
                    None,
                    env.observation_space["satellite_0"],
                    env.action_space["satellite_0"],
                    {},
                ),
                "hap": (
                    None,
                    env.observation_space["hap_0"],
                    env.action_space["hap_0"],
                    {},
                ),
                "local": (
                    None,
                    env.observation_space["local_0"],
                    env.action_space["local_0"],
                    {},
                ),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id.split("_")[0],
        )
        .training(
            lr=0.0005,
            sgd_minibatch_size=512,
            train_batch_size=2000,
            num_sgd_iter=30,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=50.0,
            entropy_coeff=0.01,
            vf_loss_coeff=1.0,
        )
        .env_runners(num_env_runners=args.num_workers, sample_timeout_s=None)
        .resources(num_gpus=args.num_gpus)
    )

    stop = {
        "training_iteration": args.num_iterations,
    }

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            name=f"tn_ntn_{args.env}",
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
            storage_path=os.path.abspath("./logs"),
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    best_checkpoint = best_result.checkpoint
    print(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on TN-NTN environment")
    parser.add_argument(
        "--env",
        type=str,
        choices=["hierarchical"],
        default="hierarchical",
        help="Environment type",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=1000, help="Number of training iterations"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of worker processes"
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=0.9,
        help="Number of GPUs to use (can be fractional)",
    )

    args = parser.parse_args()

    main(args)
