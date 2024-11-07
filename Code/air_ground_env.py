import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv

import utils
from cfg import ENV_CONFIG


class AirGroundEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.config = ENV_CONFIG if config is None else config
        self.num_haps = self.config["num_haps_per_beam"]
        self.num_regions_per_hap = self.config["num_regions_per_hap"]
        self.num_tbs_per_region = self.config["num_tbs_per_region"]
        self.num_uavs_per_region = self.config["num_uavs_per_region"]
        self.num_users_per_region = self.config["num_users_per_region"]
        self.total_bandwidth = self.config["total_bandwidth"]
        self.carrier_frequency = self.config["carrier_frequency"]
        self.num_subbands = self.config["num_subbands"]
        self.region_size = self.config["region_size"]

        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        self.power_allocations = np.zeros(
            (
                self.num_haps * self.num_regions_per_hap,
                self.num_users_per_region,
            )
        )
        self.channel_gains = self._generate_channel_gains()
        self.interference = self._generate_interference()

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.channel_gains = self._generate_channel_gains()
        self.interference = self._generate_interference()
        self.hap_allocations = np.zeros((self.num_haps, self.num_subbands))
        self.tbs_uav_allocations = np.zeros(
            (
                self.num_haps * self.num_regions_per_hap,
                self.num_tbs_per_region + self.num_uavs_per_region,
                self.num_subbands,
            )
        )
        self.user_positions = self._generate_user_positions()
        self.uav_positions = self._generate_uav_positions()
        self.power_allocations = np.zeros(
            (
                self.num_haps * self.num_regions_per_hap,
                self.num_users_per_region,
            )
        )

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
        obs_spaces = {
            "hap": spaces.Dict(
                {
                    "spectrum_allocation": spaces.Box(
                        low=0, high=np.inf, shape=(self.num_subbands,)
                    ),
                    "region_demands": spaces.Box(
                        low=0, high=np.inf, shape=(self.num_regions_per_hap,)
                    ),
                }
            ),
            "local": spaces.Dict(
                {
                    "spectrum_allocation": spaces.Box(
                        low=0, high=np.inf, shape=(self.num_subbands,)
                    ),
                    "user_positions": spaces.Box(
                        low=0,
                        high=self.region_size[0],
                        shape=(self.num_users_per_region, 2),
                    ),
                    "uav_positions": spaces.Box(
                        low=0,
                        high=self.region_size[0],
                        shape=(self.num_uavs_per_region, 2),
                    ),
                    "channel_gains": spaces.Box(
                        low=0,
                        high=np.inf,
                        shape=(
                            self.num_users_per_region,
                            self.num_tbs_per_region + self.num_uavs_per_region,
                        ),
                    ),
                }
            ),
        }
        return spaces.Dict(
            {
                f"{agent_type}_{i}": space
                for agent_type, space in obs_spaces.items()
                for i in range(self._get_num_agents(agent_type))
            }
        )

    def _create_action_space(self):
        action_spaces = {
            "hap": spaces.Box(low=0, high=1, shape=(self.num_subbands,)),
            "local": spaces.Box(
                low=0,
                high=1,
                shape=(
                    self.num_subbands
                    + self.num_users_per_region
                    + self.num_uavs_per_region * 2,
                ),
            ),
        }
        return spaces.Dict(
            {
                f"{agent_type}_{i}": space
                for agent_type, space in action_spaces.items()
                for i in range(self._get_num_agents(agent_type))
            }
        )

    def _get_num_agents(self, agent_type):
        if agent_type == "hap":
            return self.num_haps
        elif agent_type == "local":
            return self.num_haps * self.num_regions_per_hap
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def _generate_user_positions(self):
        return np.array(
            [
                utils.generate_user_positions(
                    self.num_users_per_region,
                    (0, self.region_size[0], 0, self.region_size[1]),
                )
                for _ in range(self.num_haps * self.num_regions_per_hap)
            ]
        )

    def _generate_channel_gains(self):
        return np.random.exponential(
            2.75,
            size=(
                self.num_haps * self.num_regions_per_hap,
                self.num_users_per_region,
                self.num_tbs_per_region + self.num_uavs_per_region,
            ),
        ) * (10**-7)

    def _generate_interference(self):
        return np.random.exponential(
            0.68,
            size=(
                self.num_haps * self.num_regions_per_hap,
                self.num_users_per_region,
            ),
        ) * (10**-9)

    def _generate_uav_positions(self):
        return np.array(
            [
                utils.generate_user_positions(
                    self.num_uavs_per_region,
                    (0, self.region_size[0], 0, self.region_size[1]),
                )
                for _ in range(self.num_haps * self.num_regions_per_hap)
            ]
        )

    def _get_obs(self):
        obs = {}

        for i in range(self.num_haps):
            hap_obs = {
                "spectrum_allocation": self.hap_allocations[i],
                "region_demands": self._compute_region_demands(i),
            }
            obs[f"hap_{i}"] = hap_obs

        for i in range(self.num_haps * self.num_regions_per_hap):
            local_obs = {
                "spectrum_allocation": self.tbs_uav_allocations[i].sum(axis=0),
                "user_positions": self.user_positions[i],
                "uav_positions": self.uav_positions[i],
                "channel_gains": self._compute_channel_gains(i),
            }
            obs[f"local_{i}"] = local_obs

        return obs

    def _process_actions(self, action_dict):
        for i in range(self.num_haps):
            self.hap_allocations[i] = action_dict[f"hap_{i}"].copy()
            hap_sum = np.sum(self.hap_allocations[i])
            if hap_sum > 0:
                self.hap_allocations[i] /= hap_sum
            else:
                self.hap_allocations[i] = np.ones_like(self.hap_allocations[i]) / len(
                    self.hap_allocations[i]
                )

        for i in range(self.num_haps * self.num_regions_per_hap):
            local_action = action_dict[f"local_{i}"]
            spectrum_access = local_action[: self.num_subbands]
            power_allocation = local_action[
                self.num_subbands : self.num_subbands + self.num_users_per_region
            ]
            uav_movement = local_action[
                self.num_subbands + self.num_users_per_region :
            ].reshape(self.num_uavs_per_region, 2)

            self.tbs_uav_allocations[i] = spectrum_access.copy()
            self.power_allocations[i] = power_allocation.copy()
            power_sum = np.sum(self.power_allocations[i])
            if power_sum > 0:
                self.power_allocations[i] /= power_sum
            else:
                self.power_allocations[i] = np.ones_like(
                    self.power_allocations[i]
                ) / len(self.power_allocations[i])
            self._update_uav_positions(i, uav_movement)

    def _update_uav_positions(self, region_idx, uav_movements):
        for j, movement in enumerate(uav_movements):
            self.uav_positions[region_idx][j] += movement * self.config["uav_step_size"]
            self.uav_positions[region_idx][j] = np.clip(
                self.uav_positions[region_idx][j], 0, self.region_size[0]
            )

    def _compute_region_demands(self, hap_idx):
        return np.random.uniform(0, 1, self.num_regions_per_hap)

    def _compute_channel_gains(self, region_idx):
        num_nodes = self.num_tbs_per_region + self.num_uavs_per_region
        channel_gains = np.random.exponential(
            1, size=(self.num_users_per_region, num_nodes)
        )
        return channel_gains

    def _compute_rewards(self):
        rewards = {}

        for i in range(self.num_haps):
            hap_reward = self._compute_hap_reward(i)
            rewards[f"hap_{i}"] = hap_reward

        for i in range(self.num_haps * self.num_regions_per_hap):
            local_reward = self._compute_local_reward(i)
            rewards[f"local_{i}"] = local_reward

        return rewards

    def _compute_hap_reward(self, hap_idx):
        total_spectral_efficiency = 0
        total_fairness = 0
        total_qos_violation = 0

        for i in range(self.num_regions_per_hap):
            region_idx = hap_idx * self.num_regions_per_hap + i
            spectral_efficiency, fairness, qos_violation = (
                self._compute_region_performance(region_idx)
            )
            total_spectral_efficiency += spectral_efficiency
            total_fairness += fairness
            total_qos_violation += qos_violation

        avg_spectral_efficiency = total_spectral_efficiency / self.num_regions_per_hap
        avg_fairness = total_fairness / self.num_regions_per_hap
        avg_qos_violation = total_qos_violation / self.num_regions_per_hap

        norm_spectral_efficiency = avg_spectral_efficiency / 1.5
        norm_fairness = avg_fairness
        norm_qos_violation = min(avg_qos_violation / 0.5, 1)

        hap_reward = (
            37.98 * norm_spectral_efficiency
            + 21.4 * norm_fairness
            - 33.6 * norm_qos_violation
        )

        return hap_reward

    def _compute_local_reward(self, region_idx):
        spectral_efficiency, fairness, qos_violation = self._compute_region_performance(
            region_idx
        )

        uav_penalty = sum(
            not utils.is_uav_in_region(
                pos, (0, self.region_size[0], 0, self.region_size[1])
            )
            for pos in self.uav_positions[region_idx]
        )

        max_uav_penalty = self.num_uavs_per_region
        normalized_uav_penalty = (
            uav_penalty / max_uav_penalty if max_uav_penalty > 0 else 0
        )

        norm_spectral_efficiency = spectral_efficiency / 1.5
        norm_fairness = fairness
        norm_qos_violation = min(qos_violation / 0.5, 1)

        local_reward = (
            33.11 * norm_spectral_efficiency
            + 18.76 * norm_fairness
            - 24.12 * norm_qos_violation
            - 16.07 * normalized_uav_penalty
        )

        return local_reward

    def _compute_region_performance(self, region_idx):
        spectrum_efficiency = utils.spectrum_utilization_efficiency(
            self.tbs_uav_allocations[region_idx].sum(axis=0), self.num_subbands
        )

        achievable_rates = []
        qos_violations = []
        for user in range(self.num_users_per_region):
            channel_gain = self.channel_gains[region_idx, user].max()
            interference = self.interference[region_idx, user]
            noise_power = utils.db2pow(self.config["noise_power"])

            transmit_power = (
                self.config["tbs_power"] if user % 2 == 0 else self.config["uav_power"]
            )
            transmit_power = utils.db2pow(transmit_power)

            sinr = utils.calculate_sinr(
                channel_gain,
                self.power_allocations[region_idx, user] * transmit_power,
                interference,
                (1 - self.power_allocations[region_idx, user]) * transmit_power,
                noise_power,
            )
            rate = utils.calculate_achievable_rate(
                self.total_bandwidth / self.num_subbands / 10, sinr
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
