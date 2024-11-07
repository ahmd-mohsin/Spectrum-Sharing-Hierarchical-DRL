import argparse
import os
import time
import warnings

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

import utils
from cfg import ENV_CONFIG
from hierarchical_env import HierarchicalTNNTNEnv

warnings.filterwarnings("ignore")


def create_env(env_name):
    if env_name == "hierarchical":
        return HierarchicalTNNTNEnv()
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def evaluate_agent(checkpoint_path, num_episodes=1, num_steps=500):
    ray.init(
        _temp_dir=os.path.abspath("./tmp"),
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONWARNINGS": "ignore"}},
    )

    register_env("tn_ntn_env", lambda config: create_env("hierarchical"))

    agent = Algorithm.from_checkpoint(checkpoint_path)
    env = create_env("hierarchical")

    throughputs = []
    spectral_efficiencies = []
    execution_times = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_throughput = []
        episode_spectral_efficiency = []
        episode_execution_time = []

        for step in range(num_steps):
            start_time = time.time()

            actions = {
                agent_id: agent.compute_single_action(
                    agent_obs, policy_id=agent_id.split("_")[0]
                )
                for agent_id, agent_obs in obs.items()
            }

            obs, rewards, terminated, truncated, info = env.step(actions)

            execution_time = time.time() - start_time
            episode_execution_time.append(execution_time)

            if step % 10 == 0:
                total_rate = 0
                total_users = 0
                for region in range(
                    env.num_beams * env.num_haps_per_beam * env.num_regions_per_hap
                ):
                    for user in range(env.num_users_per_region):
                        channel_gain = env.channel_gains[region, user].max()
                        interference = env.interference[region, user]
                        noise_power = utils.db2pow(ENV_CONFIG["noise_power"])
                        transmit_power = (
                            ENV_CONFIG["tbs_power"]
                            if user % 2 == 0
                            else ENV_CONFIG["uav_power"]
                        )
                        transmit_power = utils.db2pow(transmit_power)

                        sinr = utils.calculate_sinr(
                            channel_gain,
                            env.power_allocations[region, user] * transmit_power,
                            interference,
                            (1 - env.power_allocations[region, user]) * transmit_power,
                            noise_power,
                        )
                        rate = utils.calculate_achievable_rate(
                            ENV_CONFIG["total_bandwidth"]
                            / ENV_CONFIG["num_subbands"]
                            / 10,
                            sinr,
                        )
                        total_rate += rate
                        total_users += 1

                avg_throughput = total_rate / total_users
                episode_throughput.append(avg_throughput)

                spectral_efficiency = utils.calculate_spectral_efficiency(
                    total_rate, ENV_CONFIG["total_bandwidth"]
                )
                episode_spectral_efficiency.append(spectral_efficiency)

            if terminated["__all__"] or truncated["__all__"]:
                break

        throughputs.extend(episode_throughput)
        spectral_efficiencies.extend(episode_spectral_efficiency)
        execution_times.extend(episode_execution_time)

    avg_throughput = np.mean(throughputs)
    avg_spectral_efficiency = np.mean(spectral_efficiencies)
    avg_execution_time = np.mean(execution_times)

    results = {
        "avg_throughput": avg_throughput,
        "avg_spectral_efficiency": avg_spectral_efficiency,
        "avg_execution_time": avg_execution_time,
        "throughputs": throughputs,
    }

    return results


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "h_throughput.txt"), "w") as f:
        throughput_list = [float(t) for t in results["throughputs"]]
        f.write(str(throughput_list))

    with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"Average Overall Throughput (bps): {results['avg_throughput']:.4f}\n")
        f.write(
            f"Average Spectral Efficiency (bps/Hz): {results['avg_spectral_efficiency']:.4f}\n"
        )
        f.write(f"Average Execution Time (s): {results['avg_execution_time']:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO agent on TN-NTN environment"
    )
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    results = evaluate_agent(args.checkpoint_path)
    save_results(results, args.output_dir)

    print("Evaluation completed. Results saved in:", args.output_dir)
