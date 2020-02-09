using Optim
using Plots
using LineSearches
using SparseArrays
using LinearAlgebra
using IterativeSolvers

# Pass eigenequation solution to adjoint calculation.
mutable struct Schrodinger
    Ψ0
    N
    dx
    A
    E
    Ψ
end

# Solve the eigenequation.
function schrodinger_fd(V)
    N = schr.N
    dx = schr.dx
    Ψ0 = schr.Ψ0

    # Center-difference scheme.
    A = spdiagm(-1 => [1.0 for i in 1:N-1], 0 => [2.0 for i in 1:N], 1 => [1.0 for i in 1:N-1])

    # Periodic boundary conditions.
    A[1, N] = 1.0
    A[N, 1] = 1.0

    A = -A / dx^2 + Diagonal(V)
    schr.A = A

    # Smallest values.
    E = eigvals(Matrix(A))[1]
    Ψ = eigvecs(Matrix(A))[:, 1]

    # Pick sign.
    if(sum(Ψ) < 0)
        Ψ = -Ψ
    end

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
function schrodinger_fd_adj(gp, V)
    Ψ0 = schr.Ψ0
    N = schr.N
    dx = schr.dx
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
m = 0.02
x = [i for i in -1:m:1]
N = length(x)
dx = x[2] - x[1]

# Target solution, normalize and pick sign.
Ψ0 = 1.0 .+ sin.(π .* x .+ cos.(3 * π .* x))
Ψ0 = Ψ0 / sqrt(transpose(Ψ0) * Ψ0)
if(sum(Ψ0) < 0)
    Ψ0 = -Ψ0
end

# Initial guess.
V0 = [0.0 for i in -1:m:1]

A = spdiagm(-1 => [1.0 for i in 1:N-1], 0 => [2.0 for i in 1:N], 1 => [1.0 for i in 1:N-1])

schr = Schrodinger(Ψ0, N, dx, A, 0.0, zeros(N))

# Run to populate values.
schrodinger_fd(V0)

# TODO: add preconditioners from Preconditioners.jl
# https://julianlsolvers.github.io/Optim.jl/stable/#algo/precondition/
# Optimize using the conjugate gradient method and the Nocedal and Wright line search.
res= optimize(schrodinger_fd,
              schrodinger_fd_adj,
              V0,
              ConjugateGradient(;alphaguess = LineSearches.InitialStatic(),
                                linesearch = LineSearches.StrongWolfe()),
              Optim.Options(iterations = 1500,
                            show_trace = true))

show(res)
V = Optim.minimizer(res)

# Calculate the Ψ for the optimized V.
schrodinger_fd(V)
Ψ = schr.Ψ

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
