from mpi4py import MPI
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def make_gif(data, gifpath, fps=25, cmap="twilight", vmin=-0.02, vmax=0.02):
    import os
    import glob
    import imageio
    import matplotlib.pyplot as plt
    import shutil
    import tqdm
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if "temp" not in os.listdir("./"):
        os.mkdir("temp")
    zfill_param = int(np.ceil(np.log10(len(data))))

    print("Save Images...")
    for i in tqdm.tqdm(range(len(data))):
        # plt.plot(0,0,"o", color="black", s=)
        plt.plot(data[i, :, 0], data[i, :, 1], "o", label="Final positions")
        plt.xlim(-30, 100)
        plt.ylim(-30, 30)
        plt.savefig("temp/pic" + str(i).zfill(zfill_param) + ".png")
        plt.clf()
    print("Done.")

    print("Make Gif...")
    # for filename in sorted(glob.glob("temp/pic*")):
    images = [imageio.imread(filename)
              for filename in sorted(glob.glob("temp/pic*"))]
    imageio.mimsave(gifpath, images, fps=fps)
    shutil.rmtree("./temp")
    print("Done.")


def compute_force(x):
    rm = 2
    a = np.zeros(x.shape)
    r_vec = x - np.zeros(x.shape)
    r = np.linalg.norm(r_vec, axis=1)
    a = (12*rm**12/r**14 - 12*rm**6/r**8)[:, np.newaxis]*r_vec
    #a[np.where(r < 0.5*rm)] = 0
    return a


def solve_nbody(x, v, dt, steps):
    n = x.shape[0]
    traj = np.zeros((steps, n, 2))
    for s in tqdm(range(steps)):
        x = x + v*dt
        a = compute_force(x)
        v = v + a*dt
        traj[s] = x
    return traj


if __name__ == "__main__":
    # The process ID (integer 0-3 for 4-process run)
    rank = MPI.COMM_WORLD.rank
    print("Hello i am CPI ", rank)
    np.random.seed(rank)

    n = int(1e4)
    dt = 0.1
    steps = 300
    boxsize = 10
    velocity_steps = 10*4
    vmin = 0.5
    vmax = 10
    impact_parameter_max = 5
    start_position = -10

    v_array = np.linspace(vmin, vmax, velocity_steps)
    v_per_cpu = np.split(v_array, 4)

    x = np.array([start_position*np.ones(n),
                 np.linspace(0, impact_parameter_max, n)]).T

    #v = np.array([np.ones(n)*2, np.zeros(n)]).T

    f = h5py.File('/home/max/Documents/scattering_data.hdf5',
                  'w', driver='mpio', comm=MPI.COMM_WORLD)

    dset = f.create_dataset(
        'test', (4, velocity_steps//4, steps, n, 2), dtype='f')

    traj = []
    for i, v0 in enumerate(v_per_cpu[rank]):

        v = np.array([v0*np.ones(n), np.zeros(n)]).T

        traj.append(list(solve_nbody(x, v, dt, steps)))

    dset[rank] = traj

    f.close()

    traj = np.array(traj)
    print("Create Plot")

    plt.plot(x[:, 0], x[:, 1], "o", label="Initial positions")
    plt.plot(traj[-1, steps//2, :, 0], traj[-1, steps//2, :, 1],
             "o", label="Final positions")
    plt.plot(traj[-1, -1, :, 0], traj[-1, -1, :, 1],
             "o", label="Final positions")
    plt.title("Spatial distribution after {} steps".format(steps))
    plt.legend()
    plt.show()
    # print(traj[-1, :, 0].shape)

    #make_gif(traj, "duck_scattering.gif")
