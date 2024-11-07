# cfg.py

# Environment Configuration
ENV_CONFIG = {
    "num_episodes": 1000,
    "max_steps_per_episode": 500,
    "satellite_decision_interval": 50,
    "hap_decision_interval": 10,
    "tbs_uav_decision_interval": 1,
    # Network parameters
    "num_beams": 2,
    "num_haps_per_beam": 1,
    "num_regions_per_hap": 2,
    "num_tbs_per_region": 2,
    "num_uavs_per_region": 1,
    "num_users_per_region": 10,
    # Spectrum parameters
    "total_bandwidth": 200e6,
    "carrier_frequency": 28e9,
    "num_subbands": 10,
    # Node parameters (powers are in dBm)
    "satellite_altitude": 550e3,
    "hap_altitude": 20e3,
    "uav_altitude": 100,
    "satellite_power_range": (33, 45),
    "hap_power_range": (28, 36),
    "tbs_power": 16,
    "uav_power": 8,
    # Region parameters
    "region_size": (2000, 2000),  # 2x2 km region
    # Channel parameters
    "noise_power": -174,  # dBm/Hz
    # UAV movement
    "uav_step_size": 10,  # meters
    # Reward weights
    "reward_weights": {
        "spectrum_efficiency": 1.0,
        "fairness": 0.5,
        "qos_violation": -1.0,
        "uav_out_of_bounds": -0.1,
    },
}
