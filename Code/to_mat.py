import ast

import scipy.io

with open("results/es_throughput.txt", "r") as file:
    data = file.read()

data_list = ast.literal_eval(data)
scipy.io.savemat("results/es_throughput.mat", {"es_throughput": data_list})

with open("results/ra_throughput.txt", "r") as file:
    data = file.read()

data_list = ast.literal_eval(data)
scipy.io.savemat("results/ra_throughput.mat", {"ra_throughput": data_list})

with open("results/h_throughput.txt", "r") as file:
    data = file.read()

data_list = ast.literal_eval(data)
scipy.io.savemat("results/h_throughput.mat", {"h_throughput": data_list})

with open("results/mappo_throughput.txt", "r") as file:
    data = file.read()

data_list = ast.literal_eval(data)
scipy.io.savemat("results/mappo_throughput.mat", {"mappo_throughput": data_list})
