#using AbstractFFTs
using FFTW
using Plots
using Distributions

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

end

function construct(grid_size, Reynolds)
    # input data
    nx = grid_size
    ny = grid_size
    nk = nx÷2+1
    Re = Reynolds
    ReI = 0
    if Re != 0
        ReI = 1/Re
    end
    dt = 0.0001
    pad = 3/2
    time = 0
    it = 0

    fftw_num_threads = 8

    # we assume 2pi periodic domain in each dimensions
    x = range(0, 2*pi, nx+1)[1:end-1]
    dx = x[2]-x[1]
    y = range(0, 2*pi, ny+1)[1:end-1]
    dy = y[2]-y[1]

    kx = fftfreq(nx) .* nx
    ky = fftfreq(ny) .* ny

    kxs, kys = meshgrid(kx, ky[1:nk])

    k2 = kxs.^2 .+ kys.^2
    k2I = zeros(nk, ny)
    fk = k2 .!= 0.0
    k2I[fk] = 1.0 ./ k2[fk]

    mx = convert(Int, floor(pad*nx))
    mk = convert(Int, floor(pad*nk))
    my = convert(Int, floor(pad*ny))

    padder = convert(Array{Bool}, ones(Int, my))
    padder[convert(Int, ny/2+1):convert(Int, ny*(pad-0.5))] .= false

    u = zeros(nx, ny)
    v = zeros(nx, ny)
    w = zeros(nx, ny)
    f = zeros(nx, ny)

    uh  = zeros(Complex{Float64}, nk, ny)
    vh  = zeros(Complex{Float64}, nk, ny)
    wh0 = zeros(Complex{Float64}, nk, ny)
    wh  = zeros(Complex{Float64}, nk, ny)
    fh  = zeros(Complex{Float64}, nk, ny)
    psih = zeros(Complex{Float64}, nk, ny)
    dwhdt = zeros(Complex{Float64}, nk, ny)

    flow = Fluid(nx, ny, nk, Re, ReI,dt,pad, 
                time, it, fftw_num_threads, 
                x, dx, y, dy, kx, ky, fk, k2, k2I,
                mx, mk, my, padder, u, v, w, 
                f, uh, vh, wh0, wh, fh, psih, dwhdt)
    return flow
end


function get_u(self::Fluid)
    self.uh = 1im.*self.ky'.*self.psih
    self.u = irfft(self.uh, self.nx)
end

function get_v(self::Fluid)
    self.vh = -1im.*self.kx[1:self.nk] .* self.psih
    self.v = irfft(self.vh, self.nx)
end

function cfl_limit(self::Fluid)
    get_u(self)
    get_v(self)
    Dc = maximum(π .* ((1.0 .+ abs.(self.u)) ./ self.dx +
                    (1.0 .+ abs.(self.v)) ./ self.dy))
    Dmu = π^2*(self.dx^(-2) + self.dy^(-2))
    return sqrt(3.0) / (Dc + Dmu)
end

function get_psih(self::Fluid)
    self.psih =  self.wh .* self.k2I
end

function add_convection(self::Fluid)
    # padded arrays
    j1f_padded = zeros(Complex{Float64}, self.mk, self.my)
    j2f_padded = zeros(Complex{Float64}, self.mk, self.my)
    j3f_padded = zeros(Complex{Float64}, self.mk, self.my)
    j4f_padded = zeros(Complex{Float64}, self.mk, self.my)

    j1f_padded[1:self.nk, self.padder] = 1.0im .* self.kx[1:self.nk] .* self.psih
    j2f_padded[1:self.nk, self.padder] = 1.0im .* self.kx' .* self.wh
    j3f_padded[1:self.nk, self.padder] = 1.0im .* self.ky' .* self.psih
    j4f_padded[1:self.nk, self.padder] = 1.0im .* self.kx[1:self.nk] .* self.wh

    # ifft
    j1 = irfft(j1f_padded, self.mx)
    j2 = irfft(j2f_padded, self.mx)
    j3 = irfft(j3f_padded, self.mx)
    j4 = irfft(j4f_padded, self.mx)

    jacp = j1 .* j2 .- j3 .* j4

    jacpf = rfft(jacp)

    self.dwhdt = jacpf[1:self.nk, self.padder] .* self.pad^(2)  # this term is the result of padding
    #return dwhdt
end

function add_diffusion(self::Fluid)
    self.dwhdt = self.dwhdt - (self.ReI .* self.k2 .* self.wh)
    #return new_dwhdt
end


function update(self::Fluid)
    # iniitalise field
    self.wh0 = self.wh

    for k=3:-1:1
        # invert Poisson equation for the stream function (changes to k-space)
        get_psih(self)
        
        # get convective forces (resets dwhdt)
        add_convection(self)
        
        # add diffusion
        add_diffusion(self)
        
        # step in time
        self.wh = self.wh0 + (self.dt/k) .* self.dwhdt
    end
    self.time += self.dt
    self.dt = cfl_limit(self)
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
    psih = zeros(Complex{Float64}, self.nk, self.ny)
    psih[:,:] = rand(d, self.nk, self.ny).*ck 
            .+ 1.0im .* rand(d, self.nk, self.ny) .* ck

    # ṃake sure the stream function has zero mean
    cphi = 0.65*maximum(self.ky)
    wvy = sqrt.(self.k2)
    filtr = exp.(-23.6.*(wvy.-cphi).^4.)
    filtr[wvy .<= cphi] .= 1.0
    KEaux = spectral_variance(filtr.*sqrt.(self.k2).*psih, self)
    psi = psih ./ sqrt.(KEaux)

    # inverse Laplacian in k-space
    self.wh[:,:] = self.k2 .* psi

    # vorticity in physical space
    self.w[:,:] = irfft(self.wh, self.nx)
    #return field
end

function gaussians(self::Fluid)
    lin = range(1,self.nx)
    x,y = meshgrid(lin, lin)
    gaussian(A, alpha, x1, x2, y1, y2) = A*(exp.(-alpha.*((x .- x1).^2 + (y .- y1).^2)) 
                                            + exp.(-alpha.*((x .- x2).^2 + (y .- y2).^2)))

    self.w = gaussian(5, 0.01, 20, 40, 40, 20)
    self.wh = rfft(self.w)
end

function noise!(self::Fluid)
    self.w = 50.0 .*(rand(flow.nx, flow.ny).-0.5)
    self.wh = rfft(flow.w)
end

function main()
    max_iter = 10000
    println("Initilialize Field...")
    flow = construct(256, 800)

    # Initial Conditions
    #gaussians(flow)
    #noise!(flow)
    McWilliams!(flow)

    println("Done")
    println("Start Simulation...")
    for i=1:max_iter
        if i%100==0 
            println("Iteration: $i, Time: $(flow.time)")
            display(heatmap(real(irfft(flow.wh, flow.ny)), color=:seaborn_icefire_gradient, show=true))
            #display(quiver!(x[begin:5:end], y[begin:5:end], quiver=(3.0.*flow.u[begin:5:end], 3.0.*flow.v[begin:5:end]), show=true))
        end
        update(flow)
    end
    println("Finished.")
end

main()