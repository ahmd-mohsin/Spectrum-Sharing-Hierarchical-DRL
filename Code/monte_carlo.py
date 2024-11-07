import os
import time

import numpy as np

import utils
from cfg import ENV_CONFIG


def satellite_air_ground_simulation(num_samples):
    spectral_efficiencies = []
    total_throughput = 0
    execution_times = []
    throughputs = []

    for sample in range(num_samples):
        start_time = time.time()

        satellite_allocation = np.random.uniform(0, 1, ENV_CONFIG["num_beams"])
        satellite_allocation /= np.sum(satellite_allocation)
        satellite_power = np.random.uniform(*ENV_CONFIG["satellite_power_range"])

        hap_allocations = []
        hap_powers = []
        for _ in range(ENV_CONFIG["num_beams"] * ENV_CONFIG["num_haps_per_beam"]):
            hap_allocation = np.random.uniform(0, 1, ENV_CONFIG["num_subbands"])
            hap_allocation /= np.sum(hap_allocation)
            hap_allocations.append(hap_allocation)
            hap_powers.append(np.random.uniform(*ENV_CONFIG["hap_power_range"]))

        tbs_uav_allocations = []
        for _ in range(
            ENV_CONFIG["num_beams"]
            * ENV_CONFIG["num_haps_per_beam"]
            * ENV_CONFIG["num_regions_per_hap"]
        ):
            spectrum_access = np.random.randint(0, 2, ENV_CONFIG["num_subbands"])
            power_allocation = np.random.uniform(
                0, 1, ENV_CONFIG["num_users_per_region"] // 2
            )
            power_allocation /= np.sum(power_allocation)
            tbs_uav_allocations.append((spectrum_access, power_allocation))

        total_users = 0
        total_rate = 0
        for region in range(
            ENV_CONFIG["num_beams"]
            * ENV_CONFIG["num_haps_per_beam"]
            * ENV_CONFIG["num_regions_per_hap"]
        ):
            num_users = ENV_CONFIG["num_users_per_region"]
            total_users += num_users

            for user in range(num_users):
                channel_gain = np.random.exponential(0.12) * (10**-8)
                allocated_power = tbs_uav_allocations[region][1][
                    user % len(tbs_uav_allocations[region][1])
                ]
                transmit_power = (
                    ENV_CONFIG["tbs_power"]
                    if user % 2 == 0
                    else ENV_CONFIG["uav_power"]
                )
                interference = np.random.exponential(0.18) * (10**-10)
                noise_power = utils.db2pow(ENV_CONFIG["noise_power"])
                sinr = utils.calculate_sinr(
                    channel_gain,
                    allocated_power * utils.db2pow(transmit_power),
                    interference,
                    (1 - allocated_power) * utils.db2pow(transmit_power),
                    noise_power,
                )
                rate = utils.calculate_achievable_rate(
                    ENV_CONFIG["total_bandwidth"] / ENV_CONFIG["num_subbands"] / 10,
                    sinr,
                )
                total_rate += rate

        avg_throughput = total_rate / total_users
        total_throughput += avg_throughput
        throughputs.append(avg_throughput)

        spectral_efficiency = utils.calculate_spectral_efficiency(
            total_rate, ENV_CONFIG["total_bandwidth"] / ENV_CONFIG["num_subbands"]
        )
        spectral_efficiencies.append(spectral_efficiency)

        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        if (sample + 1) % 10 == 0:
            print(
                f"Step {sample + 1}: Average Throughput = {np.mean(throughputs[-10:]):.4f} bps"
            )

    return (
        spectral_efficiencies,
        total_throughput / num_samples,
        np.mean(execution_times),
        throughputs,
    )


def air_ground_simulation(num_samples):
    spectral_efficiencies = []
    total_throughput = 0
    execution_times = []
    throughputs = []

    for sample in range(num_samples):
        start_time = time.time()

        hap_allocations = []
        hap_powers = []
        for _ in range(ENV_CONFIG["num_haps_per_beam"]):
            hap_allocation = np.random.uniform(0, 1, ENV_CONFIG["num_subbands"])
            hap_allocation /= np.sum(hap_allocation)
            hap_allocations.append(hap_allocation)
            hap_powers.append(np.random.uniform(*ENV_CONFIG["hap_power_range"]))

        tbs_uav_allocations = []
        for _ in range(
            ENV_CONFIG["num_haps_per_beam"] * ENV_CONFIG["num_regions_per_hap"]
        ):
            spectrum_access = np.random.randint(0, 2, ENV_CONFIG["num_subbands"])
            power_allocation = np.random.uniform(
                0, 1, ENV_CONFIG["num_users_per_region"] // 2
            )
            power_allocation /= np.sum(power_allocation)
            tbs_uav_allocations.append((spectrum_access, power_allocation))

        total_users = 0
        total_rate = 0
        for region in range(
            ENV_CONFIG["num_haps_per_beam"] * ENV_CONFIG["num_regions_per_hap"]
        ):
            num_users = ENV_CONFIG["num_users_per_region"]
            total_users += num_users

            for user in range(num_users):
                channel_gain = np.random.exponential(2.75) * (10**-7)
                allocated_power = tbs_uav_allocations[region][1][
                    user % len(tbs_uav_allocations[region][1])
                ]
                transmit_power = (
                    ENV_CONFIG["tbs_power"]
                    if user % 2 == 0
                    else ENV_CONFIG["uav_power"]
                )
                interference = np.random.exponential(0.68) * (10**-9)
                noise_power = utils.db2pow(ENV_CONFIG["noise_power"])
                sinr = utils.calculate_sinr(
                    channel_gain,
                    allocated_power * utils.db2pow(transmit_power),
                    interference,
                    (1 - allocated_power) * utils.db2pow(transmit_power),
                    noise_power,
                )
                rate = utils.calculate_achievable_rate(
                    ENV_CONFIG["total_bandwidth"] / ENV_CONFIG["num_subbands"] / 10,
                    sinr,
                )
                total_rate += rate

        avg_throughput = total_rate / total_users
        total_throughput += avg_throughput
        throughputs.append(avg_throughput)

        spectral_efficiency = utils.calculate_spectral_efficiency(
            total_rate, ENV_CONFIG["total_bandwidth"] / ENV_CONFIG["num_subbands"]
        )
        spectral_efficiencies.append(spectral_efficiency)

        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        if (sample + 1) % 10 == 0:
            print(
                f"Step {sample + 1}: Average Throughput = {np.mean(throughputs[-10:]):.4f} bps"
            )

    return (
        spectral_efficiencies,
        total_throughput / num_samples,
        np.mean(execution_times),
        throughputs,
    )


def uav_aided_simulation(num_samples):
    spectral_efficiencies = []
    total_throughput = 0
    execution_times = []
    throughputs = []

    for sample in range(num_samples):
        start_time = time.time()

        uav_allocations = []
        for _ in range(ENV_CONFIG["num_uavs_per_region"]):
            spectrum_access = np.random.randint(0, 2, ENV_CONFIG["num_subbands"])
            power_allocation = np.random.uniform(
                0, 1, ENV_CONFIG["num_users_per_region"] // 2
            )
            power_allocation /= np.sum(power_allocation)
            uav_allocations.append((spectrum_access, power_allocation))

        total_users = 0
        total_rate = 0
        num_users = ENV_CONFIG["num_users_per_region"]
        total_users += num_users

        for user in range(num_users):
            channel_gain = np.random.exponential(3.671) * (10**-7)
            uav_index = user % ENV_CONFIG["num_uavs_per_region"]
            allocated_power = uav_allocations[uav_index][1][
                user % len(uav_allocations[uav_index][1])
            ]
            transmit_power = ENV_CONFIG["uav_power"]
            interference = np.random.exponential(0.121) * (10**-9)
            noise_power = utils.db2pow(ENV_CONFIG["noise_power"])
            sinr = utils.calculate_sinr(
                channel_gain,
                allocated_power * utils.db2pow(transmit_power),
                interference,
                (1 - allocated_power) * utils.db2pow(transmit_power),
                noise_power,
            )
            rate = utils.calculate_achievable_rate(
                ENV_CONFIG["total_bandwidth"] / ENV_CONFIG["num_subbands"] / 10, sinr
            )
            total_rate += rate

        avg_throughput = total_rate / total_users
        total_throughput += avg_throughput
        throughputs.append(avg_throughput)

        spectral_efficiency = utils.calculate_spectral_efficiency(
            total_rate,
            ENV_CONFIG["total_bandwidth"] / ENV_CONFIG["num_subbands"],
        )
        spectral_efficiencies.append(spectral_efficiency)

        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        if (sample + 1) % 10 == 0:
            print(
                f"Step {sample + 1}: Average Throughput = {np.mean(throughputs[-10:]):.4f} bps"
            )

    return (
        spectral_efficiencies,
        total_throughput / num_samples,
        np.mean(execution_times),
        throughputs,
    )


def run_monte_carlo_simulations(num_samples=500):
    print("Running Monte Carlo simulations...")

    print("\nSatellite-Air-Ground Simulation:")
    sag_efficiencies, sag_throughput, sag_time, sag_throughputs = (
        satellite_air_ground_simulation(num_samples)
    )

    print("\nAir-Ground Simulation:")
    ag_efficiencies, ag_throughput, ag_time, ag_throughputs = air_ground_simulation(
        num_samples
    )

    print("\nUAV-Aided Simulation:")
    uav_efficiencies, uav_throughput, uav_time, uav_throughputs = uav_aided_simulation(
        num_samples
    )

    sag_throughput_steps = []
    ag_throughput_steps = []
    uav_throughput_steps = []

    # evry 10 steps (results in 50 steps)
    for i in range(0, num_samples, 10):
        sag_throughput_steps.append(np.mean(sag_throughputs[i : i + 10]))
        ag_throughput_steps.append(np.mean(ag_throughputs[i : i + 10]))
        uav_throughput_steps.append(np.mean(uav_throughputs[i : i + 10]))

    os.makedirs("results", exist_ok=True)

    with open("results/monte_carlo.txt", "w") as f:
        f.write("Satellite-Air-Ground Simulation:\n")
        f.write(f"Spectral Efficiencies: {sag_efficiencies}\n")
        f.write(f"Throughputs: {sag_throughputs}\n")
        f.write(f"Execution Times: {sag_time}\n")
        f.write(f"Throughput Steps: {sag_throughput_steps}\n")

        f.write("Air-Ground Simulation:\n")
        f.write(f"Spectral Efficiencies: {ag_efficiencies}\n")
        f.write(f"Throughputs: {ag_throughputs}\n")
        f.write(f"Execution Times: {ag_time}\n")
        f.write(f"Throughput Steps: {ag_throughput_steps}\n")

        f.write("UAV-Aided Simulation:\n")
        f.write(f"Spectral Efficiencies: {uav_efficiencies}\n")
        f.write(f"Throughputs: {uav_throughputs}\n")
        f.write(f"Execution Times: {uav_time}\n")
        f.write(f"Throughput Steps: {uav_throughput_steps}\n")

    print("\nFinal Results:")
    print("\nAverage Spectral Efficiency (bps/Hz):")
    print(f"Satellite-air-ground: {np.mean(sag_efficiencies):.4f}")
    print(f"Air-ground: {np.mean(ag_efficiencies):.4f}")
    print(f"UAV-aided: {np.mean(uav_efficiencies):.4f}")

    print("\nAverage Overall Throughput (bps):")
    print(f"Satellite-air-ground: {sag_throughput:.4f}")
    print(f"Air-ground: {ag_throughput:.4f}")
    print(f"UAV-aided: {uav_throughput:.4f}")

    print("\nAverage Execution Time (s):")
    print(f"Satellite-air-ground: {sag_time:.4f}")
    print(f"Air-ground: {ag_time:.4f}")
    print(f"UAV-aided: {uav_time:.4f}")


if __name__ == "__main__":
    run_monte_carlo_simulations()
