r"""
Solves the incompressible Navier Stokes equations using the Lattice-Boltzmann
Method¹. The scenario is the flow around a cylinder in 2D which yields a van
Karman vortex street.


                                periodic
        +-------------------------------------------------------------+
        |                                                             |
        | --->                                                        |
        |                                                             |
        | --->           ****                                         |
        |              ********                                       | 
inflow  | --->        **********                                      |  outflow
        |              ********                                       |
        | --->           ****                                         |
        |                                                             |
        | --->                                                        |
        |                                                             |
        +-------------------------------------------------------------+
                                periodic

-> uniform inflow profile with only horizontal velocities at left boundary
-> outflow boundary at the right
-> top and bottom boundary connected by periodicity
-> the circle in the center (representing a slice from the 3d cylinder)
   uses a no-slip Boundary Condition
-> initially, fluid is NOT at rest and has the horizontal velocity profile
   all over the domain

¹ To be fully correct, LBM considers the compressible Navier-Stokes Equations.
This can also be seen by the fact that we have a changing macroscopic rho over
the domain and that we actively use it throughout the computations. However, our
flow speeds are below the 0.3 Mach limit which results in only minor rho
fluctuations. Hence, the fluid behaves almost incompressible. 

------

Solution strategy:

Discretize the domain into a Cartesian mesh. Each grid vertex is associated
with 9 discrete velocities (D2Q9) and 2 macroscopic velocities. Then iterate
over time.


1. Apply outflow boundary condition on the right boundary

2. Compute Macroscopic Quantities (rho and velocities)

3. Apply Inflow Profile by Zou/He Dirichlet Boundary Condition
   on the left boundary

4. Compute the discrete equilibria velocities

5. Perform a Collision step according to BGK (Bhatnagar–Gross–Krook)

6. Apply Bounce-Back Boundary Conditions on the cylinder obstacle

7. Stream alongside the lattice velocities

8. Advance in time (repeat the loop)


The 7th step implicitly yields the periodic Boundary Conditions at
the top and bottom boundary.

------

Employed Discretization:

D2Q9 grid, i.e. 2-dim space with 9 discrete
velocities per node. In Other words the 2d space is discretized into
N_x by N_y by 9 points.

    6   2   5
      \ | /
    3 - 0 - 1
      / | \
    7   4   8 

Therefore we have the shapes:

- macroscopic velocity : (N_x, N_y, 2)
- discrete velocity    : (N_x, N_y, 9)
- rho              : (N_x, N_y)


------

Lattice Boltzmann Computations

Density:

ρ = ∑ᵢ fᵢ


Velocities:

u = 1/ρ ∑ᵢ fᵢ cᵢ


Equilibrium:

fᵢᵉ = ρ Wᵢ (1 + 3 cᵢ ⋅ u + 9/2 (cᵢ ⋅ u)² − 3/2 ||u||₂²)


BGK Collision:

fᵢ ← fᵢ − ω (fᵢ − fᵢᵉ)


with the following quantities:

fᵢ  : Discrete velocities
fᵢᵉ : Equilibrium discrete velocities
ρ   : Density
∑ᵢ  : Summation over all discrete velocities
cᵢ  : Lattice Velocities
Wᵢ  : Lattice Weights
ω   : Relaxation factor

------

The flow configuration is defined using the Reynolds Number

Re = (U R) / ν

with:

Re : Reynolds Number
U  : Inflow Velocity
R  : Cylinder Radius
ν  : Kinematic Viscosity

Can be re-arranged in terms of the kinematic viscosity

ν = (U R) / Re

Then the relaxation factor is computed according to

ω = 1 / (3 ν + 0.5)

------

Note that this scheme can become unstable for Reynoldsnumbers >~ 350 ²

² Note that the stability of the D2Q9 scheme is mathematically not
linked to the Reynoldsnumber. Just use this as a reference. Stability
for this scheme is realted to the velocity magnitude.
Consequentially, the actual limiting factor is the Mach number (the
ratio between velocity magnitude and the speed of sound).

"""
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

nit = 15_000
Re = 80

nx = 300
ny = 50

cyl_x = nx // 5
cyl_y = ny // 2
R_cyl = ny // 9

u_max = 0.04

vis = True
plot_every = 100
skip_first = 0 # 5000


r"""
LBM Grid: D2Q9
    6   2   5
      \ | /
    3 - 0 - 1
      / | \
    7   4   8 
"""

nc = 9

c = np.array([
    [0,  1,  0, -1,  0,  1, -1, -1,  1, ],
    [0,  0,  1,  0, -1,  1,  1, -1, -1, ]
])

idx = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8,
])

idx_rev = np.array([
    0, 3, 4, 1, 2, 7, 8, 5, 6,
])

w = np.array([
    4/9,                        # Center Velocity [0,]
    1/9,  1/9,  1/9,  1/9,      # Axis-Aligned Velocities [1, 2, 3, 4]
    1/36, 1/36, 1/36, 1/36,     # 45 ° Velocities [5, 6, 7, 8]
])

c_right = np.array([1, 5, 8])
c_up = np.array([2, 5, 6])
c_left = np.array([3, 6, 7])
c_dwn = np.array([4, 7, 8])
c_vert = np.array([0, 2, 4])
c_hor = np.array([0, 1, 3])


def get_density(F):
    rho = np.sum(F, axis=-1)

    return rho


def get_vel(F, rho):
    u = np.einsum(
        "NMQ,dQ->NMd",
        F,
        c,
    ) / rho[..., np.newaxis]

    return u


def get_equil(u, rho):
    u_proj = np.einsum(
        "dQ,NMd->NMQ",
        c,
        u,
    )
    u_norm = np.linalg.norm(
        u,
        axis=-1,
        ord=2,
    )
    Feq = rho[..., np.newaxis]*w[np.newaxis, np.newaxis, :]*1+3 * u_proj+9/2 * u_proj**2-3/2 * u_norm[..., np.newaxis]**2

    return Feq


def main():
    jax.config.update("jax_enable_x64", True)

    kinematic_viscosity = u_max*R_cyl / Re
    relaxation_omega = 1.0 / (3.0*kinematic_viscosity+0.5)


    # Define a mesh
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Obstacle Mask: An array of the shape like X or Y, but contains True if the
    # point belongs to the obstacle and False if not
    obstacle_mask = np.sqrt(   X   -   cyl_x**2+   Y   -   cyl_y**2)<R_cyl

    velocity_profile = np.zeros((nx, ny, 2))
    velocity_profile = velocity_profile.at[:, :, 0].set(
        u_max)

    @jax.jit
    def update(discrete_velocities_prev):
        # (1) Prescribe the outflow BC on the right boundary
        discrete_velocities_prev = discrete_velocities_prev.at[-1, :, c_left].set(
            discrete_velocities_prev[-2, :, c_left]
        )

        # (2) Macroscopic Velocities
        density_prev = get_density(discrete_velocities_prev)
        macroscopic_velocities_prev = get_vel(
            discrete_velocities_prev,
            density_prev,
        )

        # (3) Prescribe Inflow Dirichlet BC using Zou/He scheme
        macroscopic_velocities_prev =\
            macroscopic_velocities_prev.at[0, 1:-1, :].set(
                velocity_profile[0, 1:-1, :]
            )
        density_prev = density_prev.at[0, :].set(
            (
                get_density(
                    discrete_velocities_prev[0, :, c_vert].T)
                +
                2 *
                get_density(discrete_velocities_prev[0, :, c_left].T)
            ) / (
                1 - macroscopic_velocities_prev[0, :, 0]
            )
        )

        # (4) Compute discrete Equilibria velocities
        Feq = get_equil(
            macroscopic_velocities_prev,
            density_prev,
        )

        # (3) Belongs to the Zou/He scheme
        discrete_velocities_prev = \
            discrete_velocities_prev.at[0, :, c_right].set(
                Feq[0, :, c_right]
            )

        # (5) Collide according to BGK
        discrete_velocities_post_collision = (
            discrete_velocities_prev
            -
            relaxation_omega
            *
            (
                discrete_velocities_prev
                -
                Feq
            )
        )

        # (6) Bounce-Back Boundary Conditions to enfore the no-slip
        for i in range(nc):
            discrete_velocities_post_collision =\
                discrete_velocities_post_collision.at[obstacle_mask, idx[i]].set(
                    discrete_velocities_prev[obstacle_mask,
                                             idx_rev[i]]
                )

        # (7) Stream alongside lattice velocities
        discrete_velocities_streamed = discrete_velocities_post_collision
        for i in range(nc):
            discrete_velocities_streamed = discrete_velocities_streamed.at[:, :, i].set(
                np.roll(
                    np.roll(
                        discrete_velocities_post_collision[:, :, i],
                        c[0, i],
                        axis=0,
                    ),
                    c[1, i],
                    axis=1,
                )
            )

        return discrete_velocities_streamed

    discrete_velocities_prev = get_equil(
        velocity_profile,
        np.ones((nx, ny)),
    )

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 6), dpi=100)

    for iteration_index in tqdm(range(nit)):
        discrete_velocities_next = update(discrete_velocities_prev)

        discrete_velocities_prev = discrete_velocities_next

        if iteration_index % plot_every == 0 and vis and iteration_index > skip_first:
            rho = get_density(discrete_velocities_next)
            u = get_vel(
                discrete_velocities_next,
                rho,
            )
            velocity_magnitude = np.linalg.norm(
                u,
                axis=-1,
                ord=2,
            )
            d_u__d_x, d_u__d_y = np.gradient(u[..., 0])
            d_v__d_x, d_v__d_y = np.gradient(u[..., 1])
            curl = (d_u__d_y - d_v__d_x)

            # Velocity Magnitude Contour Plot in the top
            plt.subplot(211)
            plt.contourf(
                X,
                Y,
                velocity_magnitude,
                levels=50,
                cmap=cmr.amber,
            )
            plt.colorbar().set_label("Velocity Magnitude")
            # plt.gca().add_patch(plt.Circle(
            #     (cyl_x, cyl_y),
            #     R_cyl,
            #     color="darkgreen",
            # ))

            # Vorticity Magnitude Contour PLot in the bottom
            plt.subplot(212)
            plt.contourf(
                X,
                Y,
                curl,
                levels=50,
                cmap=cmr.redshift,
                vmin=-0.02,
                vmax=0.02,
            )
            plt.colorbar().set_label("Vorticity Magnitude")
            # plt.gca().add_patch(plt.Circle(
            #     (cyl_x, cyl_y),
            #     R_cyl,
            #     color="darkgreen",
            # ))

            plt.draw()
            plt.pause(0.0001)
            plt.clf()

    if vis:
        plt.show()


if __name__ == "__main__":
    main()
