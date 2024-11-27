import numpy as np
import matplotlib.pyplot as plt
import shutil
import os


def plot_result(ncpu=4):
    timesteps = [str(timestep) for timestep in range(0, 10000, 50)]
    result = []
    for timestep in timesteps:
        datastring = "data/out_" + timestep + "_CPU_{}.dat"
        data_cpu = [np.loadtxt(datastring.format(str(i)))[:, 1]
                    for i in range(ncpu)]
        data = np.concatenate(data_cpu)
        result.append(data)
    return result


def plot_ducks(idx):
    timesteps = [str(timestep) for timestep in range(0, 10000, 50)]
    result = []
    for timestep in timesteps:
        datastring = "data/duck_" + timestep + ".dat"
        #data_cpu = [np.loadtxt(datastring.format(str(i)))[:,1] for i in range(ncpu)]
        #data = np.concatenate(data_cpu)
        result.append(np.loadtxt(datastring.format(idx))[0])
    return result


result = plot_result()

fig, ax = plt.subplots(1, 4, figsize=(15, 5))

for i in range(4):
    ducks = plot_ducks(i)
    ax[i].imshow(np.array(result).T, aspect="auto", origin="lower")
    ax[i].plot(ducks)
plt.show()

shutil.rmtree("./data/", )
os.mkdir("./data")
