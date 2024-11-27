#using AbstractFFTs
using FFTW
using Plots
#FFTW.set_num_threads(8)

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

function construct(grid_size)
    # input data
    nx = grid_size
    ny = grid_size
    nk = ny÷2+1
    Re = 80
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

    kxs, kys = meshgrid(kx[1:nk], ky)

    k2 = kxs.^2 .+ kys.^2
    k2I = zeros(nx, nk)
    fk = k2 .!= 0.0
    k2I[fk] = 1.0 ./ k2[fk]

    mx = convert(Int, floor(pad*nx))
    mk = convert(Int, floor(pad*nk))
    my = convert(Int, floor(pad*ny))

    padder = convert(Array{Bool}, ones(Int, mx))
    padder[convert(Int, nx/2+1):convert(Int, nx*(pad-0.5))] .= false

    u = zeros(nx, ny)
    v = zeros(nx, ny)
    w = zeros(nx, ny)
    f = zeros(nx, ny)

    uh  = zeros(Complex{Float64}, nx, nk)
    vh  = zeros(Complex{Float64}, nx, nk)
    wh0 = zeros(Complex{Float64}, nx, nk)
    wh  = zeros(Complex{Float64}, nx, nk)
    fh  = zeros(Complex{Float64}, nx, nk)
    psih = zeros(Complex{Float64}, nx, nk)
    dwhdt = zeros(Complex{Float64}, nx, nk)

    flow = Fluid(nx, ny, nk, Re, ReI,dt,pad, 
                time, it, fftw_num_threads, 
                x, dx, y, dy, kx, ky, fk, k2, k2I,
                mx, mk, my, padder, u, v, w, 
                f, uh, vh, wh0, wh, fh, psih, dwhdt)
    return flow
end



function get_u(self::Fluid)
    self.uh = 1im.*self.ky.*self.psih
    self.u = transpose(irfft(copy(transpose(self.uh)), self.ny))
end

function get_v(self::Fluid)
    self.vh = -1im.*self.kx[1:self.nk]' .* self.psih
    self.v = transpose(irfft(copy(transpose(self.vh)), self.ny))
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
    j1f_padded = zeros(Complex{Float64}, self.mx, self.mk)
    j2f_padded = zeros(Complex{Float64}, self.mx, self.mk)
    j3f_padded = zeros(Complex{Float64}, self.mx, self.mk)
    j4f_padded = zeros(Complex{Float64}, self.mx, self.mk)

    j1f_padded[self.padder, 1:self.nk] = 1.0im .* self.kx[1:self.nk]' .* self.psih
    j2f_padded[self.padder, 1:self.nk] = 1.0im .* self.kx .* self.wh
    j3f_padded[self.padder, 1:self.nk] = 1.0im .* self.ky .* self.psih
    j4f_padded[self.padder, 1:self.nk] = 1.0im .* self.kx[1:self.nk]' .* self.wh

    # ifft
    j1 = transpose(irfft(copy(transpose(j1f_padded)), self.my))
    j2 = transpose(irfft(copy(transpose(j2f_padded)), self.my))
    j3 = transpose(irfft(copy(transpose(j3f_padded)), self.my))
    j4 = transpose(irfft(copy(transpose(j4f_padded)), self.my))

    jacp = j1 .* j2 .- j3 .* j4

    jacpf = transpose(rfft(transpose(jacp)))

    dwhdt = jacpf[self.padder, 1:self.nk] .* self.pad^(2)  # this term is the result of padding
    return dwhdt
end

function add_diffusion(self::Fluid, dwhdt)
    new_dwhdt = dwhdt - (self.ReI .* self.k2 .* self.wh)
    return new_dwhdt
end


function update(self::Fluid)
    # iniitalise field
    self.wh0 = self.wh

    for k=3:-1:1
        # invert Poisson equation for the stream function (changes to k-space)
        get_psih(self)
        
        # get convective forces (resets dwhdt)
        dwhdt = add_convection(self)
        
        # add diffusion
        dwhdt = add_diffusion(self, dwhdt)
        
        # step in time
        self.wh = self.wh0 + (self.dt/k) .* dwhdt
    end
    self.time += self.dt
    self.dt = cfl_limit(self)
    self.it += 1
end

max_iter = 10000
flow = construct(64)

lin = range(1,64)
print(lin)

x,y = meshgrid(lin, lin)
#print(x)
gaussian(A, alpha, x1, x2, y1, y2) = A*(exp.(-alpha.*((x .- x1).^2 + (y .- y1).^2)) 
                                        + exp.(-alpha.*((x .- x2).^2 + (y .- y2).^2)))

#w = gaussian(1, 0.1, 5, -5, -5, 5)#
w = 50.0 .*(rand(flow.nx, flow.ny).-0.5)

flow.w = w
flow.wh = transpose(rfft(transpose(w)))


for i=1:max_iter
    if i%100==0 
        display(heatmap(real(transpose(irfft(copy(transpose(flow.wh)), flow.ny))), color=:seaborn_icefire_gradient, show=true))
        #display(quiver!(x[begin:5:end], y[begin:5:end], quiver=(3.0.*flow.u[begin:5:end], 3.0.*flow.v[begin:5:end]), show=true))
        
    end
    update(flow)
end
