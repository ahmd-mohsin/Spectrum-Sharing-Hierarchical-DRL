import numpy as np

from air_ground_env import AirGroundEnv
from hierarchical_env import HierarchicalTNNTNEnv
from uav_aided_env import UAVAidedEnv


def test_hierarchical_env(episodes=5, steps=500):
    env = HierarchicalTNNTNEnv()
    print("Hierarchical TN-NTN Environment")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        obs, _ = env.reset()
        print("\nInitial Observation:")
        for agent, agent_obs in obs.items():
            print(f"{agent}:")
            for key, value in agent_obs.items():
                print(f"  {key}: {value.shape}")

        for step in range(steps):
            action = {
                agent: space.sample() for agent, space in env.action_space.items()
            }
            next_obs, rewards, done, _, info = env.step(action)

            if done["__all__"]:
                break

            obs = next_obs

        print("\nFinal Observation:")
        for agent, agent_obs in obs.items():
            print(f"{agent}:")
            for key, value in agent_obs.items():
                print(f"  {key}: {value.shape}")

        print("\nRewards:")
        print(rewards)

        print("\nDone:")
        print(done)

        print("\nInfo:")
        print(info)


def test_air_ground_env(episodes=5, steps=500):
    env = AirGroundEnv()
    print("\nAir-Ground Environment")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        obs, _ = env.reset()
        print("\nInitial Observation:")
        for agent, agent_obs in obs.items():
            print(f"{agent}:")
            for key, value in agent_obs.items():
                print(f"  {key}: {value.shape}")

        for step in range(steps):
            action = {
                agent: space.sample() for agent, space in env.action_space.items()
            }
            next_obs, rewards, done, _, info = env.step(action)

            if done["__all__"]:
                break

            obs = next_obs

        print("\nFinal Observation:")
        for agent, agent_obs in obs.items():
            print(f"{agent}:")
            for key, value in agent_obs.items():
                print(f"  {key}: {value.shape}")

        print("\nRewards:")
        print(rewards)

        print("\nDone:")
        print(done)

        print("\nInfo:")
        print(info)


def test_uav_aided_env(episodes=5, steps=500):
    env = UAVAidedEnv()
    print("\nUAV-Aided Environment")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        obs, _ = env.reset()
        print("\nInitial Observation:")
        for agent, agent_obs in obs.items():
            print(f"{agent}:")
            for key, value in agent_obs.items():
                print(f"  {key}: {value.shape}")

        for step in range(steps):
            action = {
                agent: space.sample() for agent, space in env.action_space.items()
            }
            next_obs, rewards, done, _, info = env.step(action)

            if done["__all__"]:
                break

            obs = next_obs

        print("\nFinal Observation:")
        for agent, agent_obs in obs.items():
            print(f"{agent}:")
            for key, value in agent_obs.items():
                print(f"  {key}: {value.shape}")

        print("\nRewards:")
        print(rewards)

        print("\nDone:")
        print(done)

        print("\nInfo:")
        print(info)


if __name__ == "__main__":
    print("Testing Hierarchical TN-NTN Environment")
    test_hierarchical_env(episodes=2, steps=10)

    print("\nTesting Air-Ground Environment")
    test_air_ground_env(episodes=2, steps=10)

    print("\nTesting UAV-Aided Environment")
    test_uav_aided_env(episodes=2, steps=10)
