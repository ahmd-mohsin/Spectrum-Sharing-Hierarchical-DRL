# exhaustive_search.py

import os
import time

import numpy as np

import utils
from cfg import ENV_CONFIG


def satellite_air_ground_exhaustive_search(num_samples):
    spectral_efficiencies = []
    throughputs = []
    execution_times = []
    throughput_steps = []

    for sample in range(num_samples):
        start_time = time.time()

        best_satellite_allocation = None
        best_satellite_power = None
        best_total_rate = 0

        for satellite_power in np.linspace(*ENV_CONFIG["satellite_power_range"], 10):
            for _ in range(100):
                satellite_allocation = np.random.dirichlet(
                    np.ones(ENV_CONFIG["num_beams"])
                )

                hap_allocations = []
                hap_powers = []
                for _ in range(
                    ENV_CONFIG["num_beams"] * ENV_CONFIG["num_haps_per_beam"]
                ):
                    hap_allocation = np.random.dirichlet(
                        np.ones(ENV_CONFIG["num_subbands"])
                    )
                    hap_allocations.append(hap_allocation)
                    hap_powers.append(np.random.uniform(*ENV_CONFIG["hap_power_range"]))

                tbs_uav_allocations = []
                for _ in range(
                    ENV_CONFIG["num_beams"]
                    * ENV_CONFIG["num_haps_per_beam"]
                    * ENV_CONFIG["num_regions_per_hap"]
                ):
                    spectrum_access = np.random.randint(
                        0, 2, ENV_CONFIG["num_subbands"]
                    )
                    power_allocation = np.random.dirichlet(
                        np.ones(ENV_CONFIG["num_users_per_region"] // 2)
                    )
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
                            ENV_CONFIG["total_bandwidth"]
                            / ENV_CONFIG["num_subbands"]
                            / 10,
                            sinr,
                        )
                        total_rate += rate

                if total_rate > best_total_rate:
                    best_total_rate = total_rate
                    best_satellite_allocation = satellite_allocation
                    best_satellite_power = satellite_power

        avg_throughput = best_total_rate / total_users
        throughputs.append(avg_throughput)

        spectral_efficiency = utils.calculate_spectral_efficiency(
            best_total_rate, ENV_CONFIG["total_bandwidth"] / ENV_CONFIG["num_subbands"]
        )
        spectral_efficiencies.append(spectral_efficiency)

        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        if (sample + 1) % 10 == 0:
            throughput_steps.append(np.mean(throughputs[-10:]))
            print(
                f"Step {sample + 1}: Average Throughput = {np.mean(throughputs[-10:]):.4f} bps"
            )

    return spectral_efficiencies, throughputs, execution_times, throughput_steps


def air_ground_exhaustive_search(num_samples):
    spectral_efficiencies = []
    throughputs = []
    execution_times = []
    throughput_steps = []

    for sample in range(num_samples):
        start_time = time.time()

        best_hap_allocations = None
        best_hap_powers = None
        best_total_rate = 0

        for _ in range(100):
            hap_allocations = []
            hap_powers = []
            for _ in range(ENV_CONFIG["num_haps_per_beam"]):
                hap_allocation = np.random.dirichlet(
                    np.ones(ENV_CONFIG["num_subbands"])
                )
                hap_allocations.append(hap_allocation)
                hap_powers.append(np.random.uniform(*ENV_CONFIG["hap_power_range"]))

            tbs_uav_allocations = []
            for _ in range(
                ENV_CONFIG["num_haps_per_beam"] * ENV_CONFIG["num_regions_per_hap"]
            ):
                spectrum_access = np.random.randint(0, 2, ENV_CONFIG["num_subbands"])
                power_allocation = np.random.dirichlet(
                    np.ones(ENV_CONFIG["num_users_per_region"] // 2)
                )
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

            if total_rate > best_total_rate:
                best_total_rate = total_rate
                best_hap_allocations = hap_allocations
                best_hap_powers = hap_powers

        avg_throughput = best_total_rate / total_users
        throughputs.append(avg_throughput)

        spectral_efficiency = utils.calculate_spectral_efficiency(
            best_total_rate, ENV_CONFIG["total_bandwidth"] / ENV_CONFIG["num_subbands"]
        )
        spectral_efficiencies.append(spectral_efficiency)

        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        if (sample + 1) % 10 == 0:
            throughput_steps.append(np.mean(throughputs[-10:]))
            print(
                f"Step {sample + 1}: Average Throughput = {np.mean(throughputs[-10:]):.4f} bps"
            )

    return spectral_efficiencies, throughputs, execution_times, throughput_steps


def uav_aided_exhaustive_search(num_samples):
    spectral_efficiencies = []
    throughputs = []
    execution_times = []
    throughput_steps = []

    for sample in range(num_samples):
        start_time = time.time()

        best_uav_allocations = None
        best_total_rate = 0

        for _ in range(100):
            uav_allocations = []
            for _ in range(ENV_CONFIG["num_uavs_per_region"]):
                spectrum_access = np.random.randint(0, 2, ENV_CONFIG["num_subbands"])
                power_allocation = np.random.dirichlet(
                    np.ones(ENV_CONFIG["num_users_per_region"] // 2)
                )
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
                    ENV_CONFIG["total_bandwidth"] / ENV_CONFIG["num_subbands"] / 10,
                    sinr,
                )
                total_rate += rate

            if total_rate > best_total_rate:
                best_total_rate = total_rate
                best_uav_allocations = uav_allocations

        avg_throughput = best_total_rate / total_users
        throughputs.append(avg_throughput)

        spectral_efficiency = utils.calculate_spectral_efficiency(
            best_total_rate, ENV_CONFIG["total_bandwidth"] / ENV_CONFIG["num_subbands"]
        )
        spectral_efficiencies.append(spectral_efficiency)

        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        if (sample + 1) % 10 == 0:
            throughput_steps.append(np.mean(throughputs[-10:]))
            print(
                f"Step {sample + 1}: Average Throughput = {np.mean(throughputs[-10:]):.4f} bps"
            )

    return spectral_efficiencies, throughputs, execution_times, throughput_steps


def run_exhaustive_search(num_samples=500):
    print("Running Exhaustive Search...")

    print("\nSatellite-Air-Ground Simulation:")
    sag_efficiencies, sag_throughputs, sag_times, sag_throughput_steps = (
        satellite_air_ground_exhaustive_search(num_samples)
    )

    # print("\nAir-Ground Simulation:")
    # ag_efficiencies, ag_throughputs, ag_times, ag_throughput_steps = (
    #     air_ground_exhaustive_search(num_samples)
    # )

    # print("\nUAV-Aided Simulation:")
    # uav_efficiencies, uav_throughputs, uav_times, uav_throughput_steps = (
    #     uav_aided_exhaustive_search(num_samples)
    # )

    os.makedirs("results", exist_ok=True)
    with open("results/exhaustive_search.txt", "w") as f:
        f.write("Satellite-Air-Ground Simulation:\n")
        f.write(f"Spectral Efficiencies: {sag_efficiencies}\n")
        f.write(f"Throughputs: {sag_throughputs}\n")
        f.write(f"Execution Times: {sag_times}\n")
        f.write(f"Throughput Steps: {sag_throughput_steps}\n\n")

        # f.write("Air-Ground Simulation:\n")
        # f.write(f"Spectral Efficiencies: {ag_efficiencies}\n")
        # f.write(f"Throughputs: {ag_throughputs}\n")
        # f.write(f"Execution Times: {ag_times}\n")
        # f.write(f"Throughput Steps: {ag_throughput_steps}\n\n")

        # f.write("UAV-Aided Simulation:\n")
        # f.write(f"Spectral Efficiencies: {uav_efficiencies}\n")
        # f.write(f"Throughputs: {uav_throughputs}\n")
        # f.write(f"Execution Times: {uav_times}\n")
        # f.write(f"Throughput Steps: {uav_throughput_steps}\n\n")

    print("\nFinal Results:")
    print("\nAverage Spectral Efficiency (bps/Hz):")
    print(f"Satellite-air-ground: {np.mean(sag_efficiencies):.4f}")
    # print(f"Air-ground: {np.mean(ag_efficiencies):.4f}")
    # print(f"UAV-aided: {np.mean(uav_efficiencies):.4f}")

    print("\nAverage Overall Throughput (bps):")
    print(f"Satellite-air-ground: {np.mean(sag_throughputs):.4f}")
    # print(f"Air-ground: {np.mean(ag_throughputs):.4f}")
    # print(f"UAV-aided: {np.mean(uav_throughputs):.4f}")

    print("\nAverage Execution Time (s):")
    print(f"Satellite-air-ground: {np.mean(sag_times):.4f}")
    # print(f"Air-ground: {np.mean(ag_times):.4f}")
    # print(f"UAV-aided: {np.mean(uav_times):.4f}")

    return {
        "spectral_efficiency": {
            "satellite_air_ground": np.mean(sag_efficiencies),
            # "air_ground": np.mean(ag_efficiencies),
            # "uav_aided": np.mean(uav_efficiencies),
        },
        "throughput": {
            "satellite_air_ground": np.mean(sag_throughputs),
            # "air_ground": np.mean(ag_throughputs),
            # "uav_aided": np.mean(uav_throughputs),
        },
        "execution_time": {
            "satellite_air_ground": np.mean(sag_times),
            # "air_ground": np.mean(ag_times),
            # "uav_aided": np.mean(uav_times),
        },
    }


if __name__ == "__main__":
    results = run_exhaustive_search()
    print("\nResults:", results)
