
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

nit = 20000
Re = 150

nx = 300
ny = 100

cyl_x = nx // 5
cyl_y = ny // 2
R_cyl = ny // 9

u_max = 0.04

vis = True
plot_every = 100
skip_first = 0  # 5000

nc = 9

c = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1, ],
              [0,  0,  1,  0, -1,  1,  1, -1, -1, ]])

idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8,])

idx_rev = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6,])

w = np.array([4/9,1/9,  1/9,  1/9,  1/9, 1/36, 1/36, 1/36, 1/36])

c_right = np.array([1, 5, 8])
c_up = np.array([2, 5, 6])
c_left = np.array([3, 6, 7])
c_dwn = np.array([4, 7, 8])
c_vert = np.array([0, 2, 4])
c_hor = np.array([0, 1, 3])

def draw_duck(nx, ny):
    img = Image.open("/home/max/Dropbox/PlasmaAstro/Exercise08/duck.jpg")
    m = int(ny//2)
    m_half = m//2
    img = np.flip(np.array(img.resize((m, m), Image.Resampling.LANCZOS))[:,:,0] < 7, axis=0)
    testmat = np.zeros((nx, ny), dtype="bool")
    testmat[nx//4-m_half:(nx//4+m_half),ny//2-m_half:(ny//2+m_half)] = img.T
    return testmat

def get_density(F):
    rho = np.sum(F, axis=-1)
    return rho

def get_vel(F, rho):
    u = np.einsum("NMQ,dQ->NMd", F, c,) / rho[..., np.newaxis]
    return u

def get_equil(u, rho):
    u_proj = np.einsum("dQ,NMd->NMQ", c, u)
    u_norm = np.linalg.norm(u, axis=-1, ord=2)
    Feq = rho[..., np.newaxis]*w[np.newaxis, np.newaxis, :] * \
        (1 + 3 * u_proj+9/2 * u_proj**2-3/2 * u_norm[..., np.newaxis]**2)
    return Feq


def main():
    nu = u_max*R_cyl / Re
    omega = 1.0 / (3.0*nu+0.5)

    # Define a mesh
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Obstacle Mask
    #obstacle = np.sqrt((X - cyl_x)**2 + (Y - cyl_y)**2) < R_cyl
    #obstacle = np.zeros((nx, ny), dtype="bool")
    #obstacle[nx//4-1:nx//4+1, ny//2-10:ny//2+10] = True
    obstacle = draw_duck(nx, ny)

    u0 = np.zeros((nx, ny, 2))
    u0[:, :, 0] = u_max

    def update(F):
        # (1) Prescribe the outflow BC on the right boundary
        F[-1, :, c_left] = F[-2, :, c_left]

        # (2) Macroscopic Velocities
        rho = get_density(F)
        u = get_vel(F, rho)

        # (3) Prescribe Inflow Dirichlet BC using Zou/He scheme
        u[0, 1:-1, :] = u0[0, 1:-1, :]

        rho[0, :] = (get_density(F[0, :, c_vert].T) + 2 *
                     get_density(F[0, :, c_left].T)) / (1 - u[0, :, 0])

        # (4) Compute discrete Equilibria velocities
        Feq = get_equil(u, rho)      # (3) Belongs to the Zou/He scheme
        F[0, :, c_right] = Feq[0, :, c_right]

        # (5) Collide according to BGK
        F_after = F-omega*(F - Feq)

        # (6) Bounce-Back Boundary Conditions to enfore the no-slip
        for i in range(nc):
            F_after[obstacle, idx[i]] = F[obstacle, idx_rev[i]]

        # (7) Stream alongside lattice velocities
        F_streamed = F_after
        for i in range(nc):
            F_streamed[:, :, i] = np.roll(
                np.roll(F_after[:, :, i], c[0, i], axis=0), c[1, i], axis=1)

        return F_streamed

    F = get_equil(u0, np.random.random(size=(nx, ny))*0.1+1.0)

    fig = plt.figure(figsize=(15, 6), dpi=100)
    

    result = np.zeros((nit//plot_every, nx, ny))

    for it in tqdm(range(nit)):
        F_next = update(F)

        F = F_next

        if it % plot_every == 0 and vis and it > skip_first:
            rho = get_density(F_next)

            u = get_vel(F_next, rho)

            #velocity_magnitude = np.linalg.norm(u, axis=-1, ord=2)

            dudx, dudy = np.gradient(u[..., 0])
            dvdx, dvdy = np.gradient(u[..., 1])
            curl = (dudy - dvdx)
            curl[obstacle] = 0.0

            result[it//plot_every] = curl

            # plt.pcolormesh(X, Y, curl, cmap="twilight", vmin=-0.02, vmax=0.02)
            # # plt.colorbar().set_label("Vorticity Magnitude")

            # plt.draw()
            # plt.pause(0.0001)
            # plt.clf()

    make_gif(result, "./lbm_duck_fast.gif")
    #plt.show()

def make_gif(data, gifpath, fps=25, cmap="twilight", vmin=-0.02, vmax=0.02):
    import os
    import glob
    import imageio
    import matplotlib.pyplot as plt
    import shutil
    import tqdm
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    if not "temp" in os.listdir("./"):
        os.mkdir("temp")
    zfill_param = int(np.ceil(np.log10(len(data))))
    print("Save Images...")
    for i in range(len(data)):
        plt.imsave("temp/pic" + str(i).zfill(zfill_param) + ".png".format(i), np.flip(data[i].T, axis=0), cmap=cmap, vmin=vmin, vmax=vmax)
    print("Done.")
    images = []
    print("Make Gif...")
    for filename in sorted(glob.glob("temp/pic*")):
        images.append(imageio.imread(filename))
    imageio.mimsave(gifpath, images, fps=fps)
    shutil.rmtree("./temp")
    print("Done.")


if __name__ == "__main__":
    main()
