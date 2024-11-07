# utils.py

import numpy as np
from comyx import fading
from comyx.utils import db2pow, generate_seed


def generate_fading(link_shape, fading_type, seed_name, **kwargs):
    """Generate fading random variables.

    Usage:
    ```python
    fading_type = (
        "rician" if "satellite" in link_type or "hap" in link_type else "rayleigh"
    )
    if fading_type == "rician":
        fading = generate_fading(
            shape,
            fading_type,
            f"<unique_link_string>",
            **{"K": db2pow(<int>), "sigma": <int>},
        )
    else:
        fading = generate_fading(
            shape,
            fading_type,
            f"<unique_link_string>",
            **{"sigma": <int>},
        )
    ```
    """

    if fading_type == "rayleigh":
        return fading.get_rvs(
            link_shape, "rayleigh", seed=generate_seed(seed_name), **kwargs
        )
    elif fading_type == "rician":
        return fading.get_rvs(
            link_shape, "rician", seed=generate_seed(seed_name), **kwargs
        )
    else:
        raise NotImplementedError(f"Fading type {fading_type} is not implemented.")


def calculate_path_loss(distance, frequency, link_type):
    """Calculate path loss based on link type."""
    c = 3e8  # Speed of light in m/s
    if link_type == "satellite-hap":
        return 20 * np.log10(4 * np.pi * distance * frequency / c) + 92.45
    elif link_type == "hap-ground":
        return 20 * np.log10(4 * np.pi * distance * frequency / c)
    elif link_type == "ground-ground":
        return (
            20 * np.log10(4 * np.pi * distance * frequency / c) + 20
        )  # Additional loss


def calculate_channel_matrix(path_loss, small_scale_fading):
    """Calculate channel matrix."""
    return np.sqrt(db2pow(-path_loss)) * small_scale_fading


def calculate_channel_gain(channel_matrix):
    """Calculate channel gain."""
    return np.abs(channel_matrix) ** 2


def jains_fairness_index(data_rates):
    """Calculate Jain's fairness index."""
    data_rates = np.array(data_rates)
    if data_rates.size == 0:
        return 0
    return np.sum(data_rates) ** 2 / (len(data_rates) * np.sum(data_rates**2))


def user_association_score(rssi, load, qos_requirement):
    """Calculate user association score."""
    # Normalize RSSI, load, and QoS
    rssi_norm = (rssi - np.min(rssi)) / (np.max(rssi) - np.min(rssi))
    load_norm = 1 - (load - np.min(load)) / (np.max(load) - np.min(load))
    qos_norm = (qos_requirement - np.min(qos_requirement)) / (
        np.max(qos_requirement) - np.min(qos_requirement)
    )

    # Weighted sum (adjust weights as needed)
    return 0.5 * rssi_norm + 0.3 * load_norm + 0.2 * qos_norm


def log_normalize(data):
    """Apply log normalization to data."""
    return np.log(data + 1e-10)  # Add small constant to avoid log(0)


def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))


def generate_user_positions(num_users, region_bounds):
    """Generate random user positions within region bounds."""
    x = np.random.uniform(region_bounds[0], region_bounds[1], num_users)
    y = np.random.uniform(region_bounds[2], region_bounds[3], num_users)
    return np.column_stack((x, y))


def spectrum_utilization_efficiency(allocated_bandwidth, total_bandwidth):
    """Calculate spectrum utilization efficiency."""
    return np.sum(allocated_bandwidth) / total_bandwidth


def calculate_sinr(
    channel_gain, power_allocation, interference, counter_allocation, noise_power
):
    """
    Calculate SINR for a user in NOMA system.

    Args:
    channel_gain (float): Channel gain for the user
    power_allocation (float): Power allocation factor for the user
    interference (float): Total interference power
    counter_allocation (float): 1 - power_allocation * power_allocation
    noise_power (float): Noise power

    Returns:
    float: SINR value
    """
    signal_power = channel_gain * power_allocation
    interference_power = interference * counter_allocation
    return signal_power / (interference_power + noise_power)


def calculate_achievable_rate(bandwidth, sinr):
    """Calculate achievable data rate using Shannon's formula."""
    return bandwidth * np.log2(1 + sinr)


def calculate_spectral_efficiency(total_throughput, total_bandwidth):
    """Calculate spectral efficiency in bps/Hz"""
    return total_throughput / total_bandwidth


def calculate_comp_sinr(channel_gains, power_allocations, interference, noise_power):
    """
    Calculate SINR for a user served by CoMP.

    Args:
    channel_gains (list): List of channel gains from different transmitters
    power_allocations (list): List of power allocation factors from different transmitters
    interference (float): Total interference power
    noise_power (float): Noise power

    Returns:
    float: SINR value for CoMP
    """
    signal_power = np.sum([g * p for g, p in zip(channel_gains, power_allocations)])
    return signal_power / (interference + noise_power)


def calculate_qos_violation(achieved_rate, required_rate):
    """Calculate QoS violation."""
    return max(0, required_rate - achieved_rate)


def is_uav_in_region(uav_position, region_bounds):
    """Check if UAV is within the region bounds."""
    x, y = uav_position
    x_min, x_max, y_min, y_max = region_bounds
    return x_min <= x <= x_max and y_min <= y <= y_max
