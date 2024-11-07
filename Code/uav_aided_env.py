import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv

import utils
from cfg import ENV_CONFIG


class UAVAidedEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.config = ENV_CONFIG if config is None else config
        self.num_uavs = self.config["num_uavs_per_region"]
        self.num_users = self.config["num_users_per_region"]
        self.total_bandwidth = self.config["total_bandwidth"]
        self.carrier_frequency = self.config["carrier_frequency"]
        self.num_subbands = self.config["num_subbands"]
        self.region_size = self.config["region_size"]

        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        self.power_allocations = np.zeros((self.num_uavs, self.num_users))
        self.channel_gains = self._generate_channel_gains()
        self.interference = self._generate_interference()

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.channel_gains = self._generate_channel_gains()
        self.interference = self._generate_interference()
        self.uav_allocations = np.zeros((self.num_uavs, self.num_subbands))
        self.user_positions = self._generate_user_positions()
        self.uav_positions = self._generate_uav_positions()
        self.power_allocations = np.zeros((self.num_uavs, self.num_users))

        obs = self._get_obs()
        return obs, {}

    def step(self, action_dict):
        self.current_step += 1

        self._process_actions(action_dict)

        obs = self._get_obs()
        rewards = self._compute_rewards()
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        info = self._get_info()

        return obs, rewards, terminated, truncated, info

    def _create_observation_space(self):
        obs_space = spaces.Dict(
            {
                "spectrum_allocation": spaces.Box(
                    low=0, high=np.inf, shape=(self.num_subbands,)
                ),
                "user_positions": spaces.Box(
                    low=0,
                    high=self.region_size[0],
                    shape=(self.num_users, 2),
                ),
                "uav_positions": spaces.Box(
                    low=0,
                    high=self.region_size[0],
                    shape=(self.num_uavs, 2),
                ),
                "channel_gains": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(self.num_users, self.num_uavs),
                ),
            }
        )
        return spaces.Dict({f"uav_{i}": obs_space for i in range(self.num_uavs)})

    def _create_action_space(self):
        action_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_subbands + self.num_users + 2,),
        )
        return spaces.Dict({f"uav_{i}": action_space for i in range(self.num_uavs)})

    def _generate_user_positions(self):
        return utils.generate_user_positions(
            self.num_users,
            (0, self.region_size[0], 0, self.region_size[1]),
        )

    def _generate_channel_gains(self):
        return np.random.exponential(
            3.671,
            size=(self.num_users, self.num_uavs),
        ) * (10**-7)

    def _generate_interference(self):
        return np.random.exponential(
            0.121,
            size=(self.num_users,),
        ) * (10**-9)

    def _generate_uav_positions(self):
        return utils.generate_user_positions(
            self.num_uavs,
            (0, self.region_size[0], 0, self.region_size[1]),
        )

    def _get_obs(self):
        obs = {}
        for i in range(self.num_uavs):
            obs[f"uav_{i}"] = {
                "spectrum_allocation": self.uav_allocations[i],
                "user_positions": self.user_positions,
                "uav_positions": self.uav_positions,
                "channel_gains": self.channel_gains[:, i],
            }
        return obs

    def _process_actions(self, action_dict):
        for i in range(self.num_uavs):
            action = action_dict[f"uav_{i}"]
            spectrum_access = action[: self.num_subbands]
            power_allocation = action[self.num_subbands : -2]
            uav_movement = action[-2:]

            self.uav_allocations[i] = spectrum_access.copy()
            self.power_allocations[i] = power_allocation.copy()
            power_sum = np.sum(self.power_allocations[i])
            if power_sum > 0:
                self.power_allocations[i] /= power_sum
            else:
                self.power_allocations[i] = np.ones_like(
                    self.power_allocations[i]
                ) / len(self.power_allocations[i])
            self._update_uav_position(i, uav_movement)

    def _update_uav_position(self, uav_idx, movement):
        self.uav_positions[uav_idx] += movement * self.config["uav_step_size"]
        self.uav_positions[uav_idx] = np.clip(
            self.uav_positions[uav_idx], 0, self.region_size[0]
        )

    def _compute_rewards(self):
        rewards = {}
        for i in range(self.num_uavs):
            rewards[f"uav_{i}"] = self._compute_uav_reward(i)
        return rewards

    def _compute_uav_reward(self, uav_idx):
        spectral_efficiency, fairness, qos_violation = self._compute_performance(
            uav_idx
        )

        uav_penalty = int(
            not utils.is_uav_in_region(
                self.uav_positions[uav_idx],
                (0, self.region_size[0], 0, self.region_size[1]),
            )
        )

        norm_spectral_efficiency = spectral_efficiency / 1.5
        norm_fairness = fairness
        norm_qos_violation = min(qos_violation / 0.5, 1)

        uav_reward = (
            33.11 * norm_spectral_efficiency
            + 18.76 * norm_fairness
            - 24.12 * norm_qos_violation
            - 16.07 * uav_penalty
        )

        return uav_reward

    def _compute_performance(self, uav_idx):
        spectrum_efficiency = utils.spectrum_utilization_efficiency(
            self.uav_allocations[uav_idx], self.num_subbands
        )

        achievable_rates = []
        qos_violations = []
        for user in range(self.num_users):
            channel_gain = self.channel_gains[user, uav_idx]
            interference = self.interference[user]
            noise_power = utils.db2pow(self.config["noise_power"])

            transmit_power = utils.db2pow(self.config["uav_power"])

            sinr = utils.calculate_sinr(
                channel_gain,
                self.power_allocations[uav_idx, user] * transmit_power,
                interference,
                (1 - self.power_allocations[uav_idx, user]) * transmit_power,
                noise_power,
            )
            rate = utils.calculate_achievable_rate(
                self.total_bandwidth / self.num_subbands / self.num_uavs, sinr
            )
            achievable_rates.append(rate)

            qos_requirement = 6e6
            qos_violations.append(utils.calculate_qos_violation(rate, qos_requirement))

        fairness = utils.jains_fairness_index(achievable_rates)
        avg_qos_violation = np.mean(qos_violations) / qos_requirement

        return spectrum_efficiency, fairness, avg_qos_violation

    def _check_terminated(self):
        terminated = {agent_id: False for agent_id in self.observation_space.keys()}
        terminated["__all__"] = False
        return terminated

    def _check_truncated(self):
        is_last_step = self.current_step >= self.config["max_steps_per_episode"]
        truncated = {
            agent_id: is_last_step for agent_id in self.observation_space.keys()
        }
        truncated["__all__"] = is_last_step
        return truncated

    def _check_done(self):
        is_last_step = self.current_step >= self.config["max_steps_per_episode"]
        return {"__all__": is_last_step}

    def _get_info(self):
        return {}
