# Hierarchical Deep Reinforcement Learning for Adaptive Resource Management in Integrated Terrestrial and Non-Terrestrial Networks

## Project Overview
This project presents a Hierarchical Deep Reinforcement Learning (HDRL) framework designed to efficiently manage spectrum sharing in an integrated terrestrial and non-terrestrial network (TN-NTN) environment. The framework enables dynamic spectrum allocation, which is increasingly vital as the number of connected devices grows with 6G technologies. Leveraging HDRL, our model learns to allocate spectrum in real-time, optimizing for both terrestrial and non-terrestrial networks, including satellite-based networks such as Starlink, to reduce interference and increase efficiency based on regional demands.

## Key Files and Structure

### Main Code Files
The main code for this project is organized within the `code/` directory. Key files are as follows:

- **`air_ground_env.py`**: Contains the environment setup and simulation functions for a terrestrial-only (air-ground) network.
- **`cfg.py`**: Configuration file that stores parameters and settings used across various modules.
- **`demo_envs.py`**: Initializes and sets up demonstration environments for testing the HDRL algorithms.
- **`eval.py`**: Evaluates the performance of trained agents and generates results for analysis.
- **`exhaustive_search.py`**: Provides a brute-force approach for baseline spectrum allocation, allowing for comparisons with HDRL results.
- **`hierarchical_env.py`**: The primary environment for hierarchical reinforcement learning, integrating terrestrial and non-terrestrial components.
- **`main.py`**: The main entry point for running training and evaluation for the HDRL agents.
- **`monte_carlo.py`**: Implements a Monte Carlo simulation approach as an alternative benchmark for spectrum allocation.
- **`to_mat.py`**: Converts simulation data to `.mat` files for further analysis in MATLAB.
- **`uav_aided_env.py`**: Defines an environment setup for UAV-aided (Unmanned Aerial Vehicles) spectrum sharing scenarios.
- **`utils.py`**: Utility functions used across different modules for general support tasks, such as logging and configuration loading.

### Supporting Files
- **`LICENSE`**: Specifies the licensing terms for this project.
- **`README.md`**: Provides an overview, instructions, and documentation for the project.

## Problem Statement
Efficient spectrum allocation is crucial due to the growing number of wireless-connected devices, which is expected to increase with the advent of 6G. With the advancement of satellite networks such as SpaceX's Starlink, non-terrestrial networks (NTNs) now have the potential to work alongside terrestrial networks (TNs) to allocate spectrum based on regional demands. However, traditional spectrum-sharing techniques focus primarily on TNs and do not scale well to TN-NTN integration.

Our work addresses this gap by using a **Hierarchical Deep Reinforcement Learning (HDRL)** approach to manage spectrum allocation across both TNs and NTNs. This approach allows DRL agents at each network layer to dynamically learn and allocate spectrum according to regional demand trends, improving adaptability and efficiency.

## Methodology
The HDRL framework operates in a layered structure, where each layer (terrestrial and non-terrestrial) consists of DRL agents that manage spectrum allocation in a decentralized manner. The agents are trained to:
- Minimize interference between users.
- Optimize power allocation.
- Adapt to real-time changes in user demand across different regions.

Through HDRL, our model aims to enhance both adaptability and efficiency in spectrum sharing, thus better supporting the rapid growth of connected devices and data usage in TN-NTN integrated environments.

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries listed in `requirements.txt`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/ahmd-moshin/Spectrum-Sharing-Hierarchical-DRL.git
   cd Spectrum-Sharing-Hierarchical-DRL
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
To train and evaluate the HDRL model, use the following command:
```bash
python code/main.py
```

For detailed configuration, modify parameters in `cfg.py`.

### Evaluation
Run `eval.py` to generate performance metrics and compare HDRL results with baseline approaches (e.g., `exhaustive_search.py`, `monte_carlo.py`).

## Results and Analysis
Results from the HDRL model will be outputted in the `results/` folder and can be visualized using `to_mat.py` to convert them for analysis in MATLAB or other visualization tools.

## License
This project is licensed under the terms specified in the `LICENSE` file.
