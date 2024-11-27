using FFTW
using Plots
using CUDA
using Distributions
CUDA.allowscalar(true)

function meshgrid(x_,y_)
    lenx = length(y_)
    leny = length(x_)
    x = x_'.* ones(lenx)
    y = ones(leny)'.* y_
    return x,y
end

mutable struct Fluid
    nx::Int
    ny::Int
    nk::Int
    Re::Int
    ReI::Float64
    dt::Float64
    pad::Float64
    time::Float64
    it::Int
    fftw_num_threads
    x 
    dx
    y 
    dy
    kx
    ky
    fk
    k2
    k2I
    mx
    mk
    my
    padder 
    u
    v
    w
    f
    uh 
    vh 
    wh0
    wh 
    fh 
    psih 
    dwhdt
    j1f_padded
    j2f_padded
    j3f_padded
    j4f_padded
    j1
    j2
    j3
    j4
    jacp
    jacpf
end

function construct(grid_size, Reynolds)
    # input data
    nx = cu(grid_size)
    ny = cu(grid_size)
    nk = cu(nx÷2+1)
    Re = Reynolds
    ReI = 0.0
    if Re != 0.0
        ReI = 1.0 / Re
    end
    dt = 0.0001
    pad = 3/2
    time = 0.0
    it = 0

    fftw_num_threads = 8

    # we assume 2pi periodic domain in each dimensions
    x = cu(range(0, 2*pi, nx+1)[1:end-1])
    dx = x[2]-x[1]
    y = cu(range(0, 2*pi, ny+1)[1:end-1])
    dy = y[2]-y[1]

    kx = cu(fftfreq(nx) .* nx)
    ky = cu(fftfreq(ny) .* ny)

    kxs, kys = meshgrid(kx, ky[1:nk])

    k2 = kxs.^2 .+ kys.^2
    k2I = zeros(Complex{Float64}, nk, ny)
    fk = k2 .!= 0.0
    k2I[fk] = 1.0 ./ k2[fk]
    k2I = cu(k2I)
    k2 = cu(k2)

    mx = convert(Int, floor(pad*nx))
    mk = convert(Int, floor(pad*nk))
    my = convert(Int, floor(pad*ny))

    padder = convert(Array{Bool}, ones(Int, my))
    padder[convert(Int, ny/2+1):convert(Int, ny*(pad-0.5))] .= false
    padder = cu(padder)

    u = cu(zeros(nx, ny))
    v = cu(zeros(nx, ny))
    w = cu(zeros(nx, ny))
    f = cu(zeros(nx, ny))

    uh  = cu(zeros(Complex{Float64}, nk, ny))
    vh  = cu(zeros(Complex{Float64}, nk, ny))
    wh0 = cu(zeros(Complex{Float64}, nk, ny))
    wh  = cu(zeros(Complex{Float64}, nk, ny))
    fh  = cu(zeros(Complex{Float64}, nk, ny))
    psih = cu(zeros(Complex{Float64}, nk, ny))
    dwhdt = cu(zeros(Complex{Float64}, nk, ny))

    # padded arrays
    j1f_padded = cu(zeros(Complex{Float64}, mk, my))
    j2f_padded = cu(zeros(Complex{Float64}, mk, my))
    j3f_padded = cu(zeros(Complex{Float64}, mk, my))
    j4f_padded = cu(zeros(Complex{Float64}, mk, my))

    j1 = cu(zeros(Float64, mx, my))
    j2 = cu(zeros(Float64, mx, my))
    j3 = cu(zeros(Float64, mx, my))
    j4 = cu(zeros(Float64, mx, my))

    jacp = cu(zeros(Float64, mx, my)) 
    jacpf= cu(zeros(Complex{Float64}, mk, my))

    flow = Fluid(nx, ny, nk, Re, ReI,dt,pad, 
                time, it, fftw_num_threads, 
                x, dx, y, dy, kx, ky, fk, k2, k2I,
                mx, mk, my, padder, u, v, w, 
                f, uh, vh, wh0, wh, fh, psih, dwhdt,
                j1f_padded, j2f_padded, j3f_padded, j4f_padded,
                j1, j2, j3, j4, jacp, jacpf)
    return flow
end

function get_u!(self::Fluid)
    self.uh[:,:] = 1im.*self.ky'.*self.psih
    self.u[:,:] = irfft(self.uh, self.ny)
end

function get_v!(self::Fluid)
    tmp = @view self.kx[1:self.nk]
    self.vh[:,:] = -1im.*tmp .* self.psih
    self.v[:,:] = irfft(self.vh, self.ny)
end

function cfl_limit!(self::Fluid)
    get_u!(self)
    get_v!(self)
    Dc = maximum(π .* ((1.0 .+ abs.(self.u)) ./ self.dx +
                    (1.0 .+ abs.(self.v)) ./ self.dy))
    Dmu = π^2*(self.dx^(-2) + self.dy^(-2))
    return sqrt(3.0) / (Dc + Dmu)
end

function get_psih!(self::Fluid)
    self.psih[:,:] =  self.wh .* self.k2I
end

function add_convection!(self::Fluid)
    tmp = @view self.kx[1:self.nk]
    self.j1f_padded[1:self.nk, self.padder] = 1.0im .* tmp .* self.psih
    self.j2f_padded[1:self.nk, self.padder] = 1.0im .* self.kx' .* self.wh
    self.j3f_padded[1:self.nk, self.padder] = 1.0im .* self.ky' .* self.psih
    self.j4f_padded[1:self.nk, self.padder] = 1.0im .* tmp .* self.wh

    # ifft
    self.j1 = irfft(self.j1f_padded, self.mx)
    self.j2 = irfft(self.j2f_padded, self.mx)
    self.j3 = irfft(self.j3f_padded, self.mx)
    self.j4 = irfft(self.j4f_padded, self.mx)

    self.jacp = self.j1 .* self.j2 .- self.j3 .* self.j4

    self.jacpf = rfft(self.jacp)

    self.dwhdt[:,:] = self.jacpf[1:self.nk, self.padder] .* self.pad^(2)  # this term is the result of padding
end

function add_diffusion!(self::Fluid)
    self.dwhdt[:,:] = self.dwhdt - (self.ReI .* self.k2 .* self.wh)
end


function update!(self::Fluid)
    # iniitalise field
    self.wh0 = self.wh

    for k=3:-1:1
        # invert Poisson equation for the stream function (changes to k-space)
        get_psih!(self)
        
        # get convective forces (resets dwhdt)
        add_convection!(self)
        
        # add diffusion
        add_diffusion!(self)
        
        # step in time
        self.wh = self.wh0 + (self.dt/k) .* self.dwhdt
    end
    self.time += self.dt
    self.dt = cfl_limit!(self)
    self.it += 1
end

function spectral_variance(x, self::Fluid)
    var_dens = 2 * abs.(x).^2 ./ (self.nx*self.ny)^2
    var_dens[:,begin] ./= 2.0
    var_dens[:,end] ./= 2.0
    return sum(var_dens)
end

function McWilliams!(self::Fluid)
    # ensemble variance proportional to the prescribed scalar wavenumber function
    ck = zeros(self.nk, self.ny)
    ck[self.fk] = (sqrt.(self.k2[self.fk]).*(1.0 .+(self.k2[self.fk]./36).^2)).^(-1)

    # Gaussian random realization for each of the Fourier components of psi
    d = Normal(0, 1)
    psih = cu(zeros(Complex{Float64}, self.nk, self.ny))
    psih[:,:] = rand(d, self.nk, self.ny).*ck 
            .+ 1.0im .* rand(d, self.nk, self.ny) .* ck

    # ṃake sure the stream function has zero mean
    cphi = 0.65*maximum(self.ky)
    wvy = sqrt.(self.k2)
    filtr = exp.(-23.6.*(wvy.-cphi).^4.)
    filtr[wvy .<= cphi] .= 1.0
    filtr = cu(filtr)
    KEaux = spectral_variance(filtr.*sqrt.(self.k2).*psih, self)
    psi = psih ./ sqrt.(KEaux)

    # inverse Laplacian in k-space
    self.wh[:,:] = cu(self.k2 .* psi)

    # vorticity in physical space
    self.w[:,:] = cu(irfft(self.wh, self.nx))
end

function gaussians(self::Fluid)
    lin = range(1,self.nx)
    x,y = meshgrid(lin, lin)
    gaussian(A, alpha, x1, x2, y1, y2) = A*(exp.(-alpha.*((x .- x1).^2 + (y .- y1).^2)) 
                                            + exp.(-alpha.*((x .- x2).^2 + (y .- y2).^2)))

    self.w[:,:] = cu(gaussian(5, 0.005, 0.33*self.nx, 0.66*self.nx, 0.66*self.nx, 0.33*self.nx))
    self.wh[:,:] = cu(rfft(self.w))
end

function noise!(self::Fluid)
    self.w[:,:] = cu(50.0 .*(rand(flow.nx, flow.ny).-0.5))
    self.wh[:,:] = cu(rfft(flow.w))
end

function shearflow!(self::Fluid)
    A = 100.0
    a = 10.0
    oned_array = range(-10,10, self.nx)
    oned_gaussians = A.*exp.(-a.*(oned_array .- 5).^2) - A.*exp.(-a.*(oned_array .+ 5).^2)
    w = oned_gaussians.*ones(self.nx)'
    w .+= rand(self.nx, self.ny).*10.0
    self.w[:,:] = cu(w)
    self.wh[:,:] = cu(rfft(self.w))
end

function main()
    max_iter = 10000
    plot_every = 100
    grid_size = 64
    Reynolds = 50
    println("Initilialize Field...")
    flow = construct(grid_size, Reynolds)

    # Initial Conditions
    gaussians(flow)
    #noise!(flow)
    #McWilliams!(flow)
    #shearflow!(flow)


    println("Done.")
    println("Start Simulation...")
    result = zeros(max_iter÷plot_every, flow.nx, flow.ny)
    for i=1:max_iter
        if i%plot_every==0 
            println("Iteration: $i, Time: $(flow.time)")
            result[(i÷plot_every),:,:] = real(irfft(Array(flow.wh), flow.nx))
            display(heatmap(result[(i÷plot_every),:,:], color=:seaborn_icefire_gradient))
            #display(quiver!(x[begin:5:end], y[begin:5:end], quiver=(3.0.*flow.u[begin:5:end], 3.0.*flow.v[begin:5:end]), show=true))
        end
        update!(flow)
    end
    println("Done.")

    println("Create Animation...")
    anim = @animate for i ∈ 1:max_iter÷plot_every
        heatmap(result[i,:,:], color=:seaborn_icefire_gradient)
    end
    gif(anim, "navier_stokes_julia_HD4.gif", fps = 15)
    println("Done.")
end

main()

