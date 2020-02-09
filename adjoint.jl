using Optim
using Plots
using LineSearches
using SparseArrays
using LinearAlgebra
using IterativeSolvers

# Pass eigenequation solution to adjoint calculation.
mutable struct Schrodinger
    A
    E
    Ψ
end

# Solve the eigenequation.
function schrodinger_fd(V, schr, p)
    N, _, Ψ0, Mesh = p

    A = Mesh + Diagonal(V)

    # Smallest values.
    E = eigvals(Matrix(A))[1]
    Ψ = eigvecs(Matrix(A))[:, 1]

    # Pick sign.
    if(sum(Ψ) < 0)
        Ψ = -Ψ
    end

    schr.A = A
    schr.E = E
    schr.Ψ = Ψ

    # Least-square error.
    err = 0.0
    for i in 1:N
        err += (Ψ[i] - Ψ0[i])^2
    end
    return err
end

# Adjoint to get the gradient.
function schrodinger_fd_adj(gp, V, schr, p)
    N, dx, Ψ0, _ = p

    A = schr.A
    E = schr.E
    Ψ = schr.Ψ

    gΨ = Ψ - Ψ0
    g = transpose(gΨ) * gΨ * dx
    gΨ = gΨ * 2 * dx

    P(Ψx) = Ψx - Ψ * (transpose(Ψ) * Ψx)
    λ = cg(A - Diagonal([E for i in 1:N]), P(gΨ))
    λ = P(λ)

    copyto!(gp, -real(conj(λ) .* Ψ))
end

# Domain.
const m = 0.02
const x = [i for i in -1:m:1]
const N = length(x)
const dx = x[2] - x[1]

# Target solution, normalize and pick sign.
Ψ0 = 1.0 .+ sin.(π .* x .+ cos.(3 * π .* x))
Ψ0 = Ψ0 / sqrt(transpose(Ψ0) * Ψ0)
if(sum(Ψ0) < 0)
    Ψ0 = -Ψ0
end

# Initial guess.
const V0 = [0.0 for i in -1:m:1]

# Center-difference scheme.
Mesh = spdiagm(-1 => [1.0 for i in 1:N-1],
               0 => [2.0 for i in 1:N],
               1 => [1.0 for i in 1:N-1])

# Periodic boundary conditions.
Mesh[1, N] = 1.0
Mesh[N, 1] = 1.0
Mesh = -Mesh / dx^2

# Build constants.
const p = [N, dx, Ψ0, Mesh]

# Build struct.
schr = Schrodinger(Mesh, 0.0, zeros(N))

# Run to populate values.
schrodinger_fd(V0, schr, p)

# TODO: add preconditioners from Preconditioners.jl
# https://julianlsolvers.github.io/Optim.jl/stable/#algo/precondition/
# Optimize using the conjugate gradient method and the Nocedal and Wright line search.
res= optimize(V0 -> schrodinger_fd(V0, schr, p),
              (gp, V) -> schrodinger_fd_adj(gp, V, schr, p),
              V0,
              ConjugateGradient(;alphaguess = LineSearches.InitialStatic(),
                                linesearch = LineSearches.StrongWolfe()),
              Optim.Options(iterations = 1500,
                            show_trace = true))

show(res)
V = Optim.minimizer(res)

# Calculate the Ψ for the optimized V.
schrodinger_fd(V, schr, p)
Ψ = schr.Ψ

# Plots results.
scatter(x, Ψ, label = "\\Psi_i",
        markersize = 3,
        markerstrokecolor = :blue,
        markercolor = :white)

plot!(x, Ψ0, label = "\\Psi_0",
      color = :red)

plot!(x, V / 1000, label = "V/1000",
      linestyle = :dash,
      color = :black)

plot!(xlims = (-1, 1),
      xticks = -1:0.2:1,
      ylims = (-0.15, 0.2),
      yticks = -0.15:0.05:0.2,
      grid = false,
      legend = :bottomleft,
      fmt = :svg)

savefig("psi1.svg")
