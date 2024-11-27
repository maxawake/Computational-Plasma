using Images
using Plots
using ProgressBars
using Einsum
using LinearAlgebra

function meshgrid(x_,y_)
    lenx = length(y_)
    leny = length(x_)
    x = x_'.* ones(lenx)
    y = ones(leny)'.* y_
    return x,y
end

function draw_duck(nx, ny)
  img = load("/home/max/Dropbox/PlasmaAstro/Exercise08/LatticeBoltzmann/duck.jpg")
  m = ny÷2
  imgg = Gray.(imresize(img, (m, m)))
  mat = transpose(convert(Array{Float64}, imgg))
  testmat = zeros(nx, ny)
  mask = mat .== 0.0
  m_half = m÷2
  testmat[nx÷4-m_half:(nx÷4+m_half-1),ny÷2-m_half:(ny÷2+m_half-1)] .= reverse(mask, dims=(2))
  return convert(BitArray, testmat)
end

function get_density!(F)
  return sum(F,dims=(3))
end

function get_vel!(F, c, ρ)
  #vel = zeros(nx, ny)
  @einsum vel[i, j, k] := F[i,j,l] * c[k, l] / ρ[i,j]
  return vel
end

function get_initial_conditions()
  # Initial Conditions - flow to the right with some perturbations
  # F = ones(nx,ny,NL) #.+ 0.01 .* rand(nx,ny,NL)
  # F[:,:,4] .*=  0.04#.* (1.0.+0.2.*cos.(2*π.*X./nx.*4))
  # ρ = get_density!(F)
  # for i in idxs
  #     F[:,:,i] .*= ρ_0 ./ ρ
  # end
  u_0 = zeros(nx,ny,2)
  u_0[:,:,1] .= u_max
  ρ = ones(nx, ny)
  F = get_equil!(u_0, ρ)  
  return F, ρ
end

function get_equil!(u, ρ)
  Feq = zeros(nx, ny, NL)
  #u_proj = zeros(nx, ny, 2)
  u_norm = zeros(nx, ny, 1)
  @einsum u_proj[i,j,k] :=  u[i,j,l] * c[l,k] 
  @einsum u_norm[i,j] = u[i,j,k] * u[i,j,k]
  # for (i, cx, cy, w) in zip(idxs, cxs, cys, weights)
  #   #println(size(Feq[:,:,i]), size(ρ), size(w), size(ux), size(uy))
  #   Feq[:,:,i] = ρ.*w.* (1.0 .+ 3.0 .*(cx.*ux.+cy.*uy) .+ 9.0 .*(cx.*ux.+cy.*uy).^2 ./2.0 .- 3.0 .*(ux.^2 .+ uy.^2)./2.0)
  # end
  lhs = (1.0 .+ 3.0 .* u_proj .+ 9.0/2.0 .* u_proj.^2 .- 3.0/2.0 .* u_norm) 
  @einsum Feq[i,j,k] := ρ[i,j] * weights[k] * lhs[i,j,k]
  return Feq
end

function get_vorticity!(ux, uy)
  dudx = (circshift(ux, (-1,0)) - circshift(ux, (1,0)))
  dudy = (circshift(uy, (0,-1)) - circshift(uy, (0,1)))
  ω = dudy - dudx
  return ω
end


# Simulation parameters
nx          = 300    # resolution x-dir
ny          = 50    # resolution y-dir
ρ_0        = 1    # average density
Re        = 80

τ         = 2.5 # collision timescale
Nt          = 15000  # number of timesteps
plot_every  = 10
u_max      = 0.04

kin_vis = (u_max * ny/9) / Re
ω = 1.0 / (3.0*kin_vis+0.5)
τ = 1.0 / ω

result = zeros(nx,ny,Nt)

# Lattice speeds / weights
NL = 9
idxs = range(1,NL)
rev_idxs = [1, 4, 3, 2, 3, 8, 9, 6, 7]
cx = [0,  1,  0, -1,  0,  1, -1, -1,  1, ]
cy = [0,  0,  1,  0, -1,  1,  1, -1, -1, ]
c = [cx cy]'
println(size(c))
weights = [4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36] # sums to 1
u_left = [4, 7, 8]
u_right = [2, 6, 9]
u_vert = [1, 3, 5]
u_hor = [1, 2, 4]



F, ρ = get_initial_conditions()


# Cylinder boundary
X, Y = meshgrid(range(0,nx-1,nx), range(0,ny-1,ny))
obstacle = (X .- nx/5).^2 .+ (Y .- ny/2).^2 .< (ny/9)^2
obstacle = obstacle'
#obstacle = draw_duck(nx, ny)

println("Start Simulation")
# Simulation Main Loop
for it=ProgressBar(1:Nt)
  # Outflow on the right
  F[end,:,u_left] = F[end-1,:, u_left]

  # Calculate fluid variables
  ρ[:,:] = get_density!(F)
  u  = get_vel!(F, c, ρ)#permutedims(sum(permutedims(F, (3,2,1)).*cxs,dims=(1)), (3,2,1)) ./ ρ
  #uy  = get_vel!(F, cys, ρ)#permutedims(sum(permutedims(F, (3,2,1)).*cys,dims=(1)), (3,2,1)) ./ ρ
  
  # (3) Prescribe Inflow Dirichlet BC using Zou/He scheme
  u[1, 2:end, :] .= u_max
  print(F[1, :, u_vert])
  rho_vert = sum(copy(transpose(F[1, :, u_vert])),2) 
  rho_left = sum(copy(transpose(F[1, :, u_left])), 2)
  ρ[1,:] = (rho_vert.+ 2.0 .* rho_left) ./ (1.0 .- u[1, :,1])

  
  # Apply Collision
  Feq = get_equil!(u, ρ)

  # Hou/He
  F[1,:,u_left] = Feq[1,:,u_left]

  F .-= (1.0/τ) .* (F .- Feq)

  # Set reflective boundaries
  bndryF = F[obstacle,:]
  bndryF = bndryF[:,rev_idxs,:]

  
  # Apply boundary 
  F[obstacle,:] = bndryF

  # Drift
  for i in idxs
    F[:,:,i] = circshift(circshift(F[:,:,i], (c[1,i], 0)), (0, c[2,i]))
    #F[:,:,i] = circshift(F[:,:,i], (cx, 0))
  end

  # Plot vorticity every nth step
  if it % 100 == 0
    w = (circshift(u[:,:,1], (-1,0)) - circshift(u[:,:,1], (1,0))) - (circshift(u[:,:,2], (0,-1)) - circshift(u[:,:,2], (0,1)))
    w[obstacle,:] .= 1.0
    #result[:,:,(it÷plot_every)] = w[:,:,1]
    display(heatmap(transpose(w[:,:,1]), aspect_ratio=:equal, color=:seaborn_icefire_gradient, clim=(-0.01, 0.01)))
    #display(quiver(X[begin:10:end], Y[begin:10:end], quiver=(ux[:,:,1][begin:10:end], uy[:,:,1][begin:10:end])))
  end
  
end

anim = @animate for i ∈ 1:Nt÷plot_every
  heatmap(result[:,:,i], color=:seaborn_icefire_gradient, clim=(-0.1, 0.1))
end
gif(anim, "LBM_duck_SD2.gif", fps = 15)
println("Done.")
# TODO: ZoeHu Boundary conditions
