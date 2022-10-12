include("../cme.jl")
include("../likelihoods.jl")

include("afl.jl")

using MomentClosure
using ProgressMeter
using OrdinaryDiffEq
using JLD2

# Binding rates for which we want to compute the KL divergence
σ_bs = [ 0.1, 0.2, 0.4, 0.7, 0.9, 1.3, 1.5, (2:0.1:3.5)..., 4.0, 4.5, 5, 6, 8, 10 ]
tmax = 1e5

ps_red = zeros(length(σ_bs))

means_lma = similar(ps_red)
stds_lma = similar(ps_red)

means_full = similar(ps_red)
stds_full = similar(ps_red)

means_red = similar(ps_red)
stds_red = similar(ps_red)

## Compute moments for the LMA
rs_nonlin = @reaction_network begin
    0.3 * (2/9), G_u --> G_u + P
    105. * (2/9), G_b --> G_b + P
    1.0, P --> 0
    σ_b, G_u + P --> G_b + P
    400, G_b --> G_u + P
end σ_b

LMA_eqs, effective_params = linear_mapping_approximation(rs_nonlin, sys_red, [1, 3], 2)
u0map = deterministic_IC([1, 0, 0], LMA_eqs)
prob_LMA = ODEProblem(LMA_eqs, u0map, tmax, [2.5])

# This looks hacky - find out how to extract the protein's first and second moments
# from the equations
idx_m = findfirst(x -> isequal(x, LMA_eqs.μ[(0,1,0)]), states(LMA_eqs.odes))
idx_m2 = findfirst(x -> isequal(x, LMA_eqs.μ[(0,2,0)]), states(LMA_eqs.odes))

@showprogress for (j, σ_b) in enumerate(σ_bs)
    prob = remake(prob_LMA, p=[σ_b])
    sol_LMA = solve(prob, KenCarp4(), saveat=[tmax])
    
    means_lma[j] = sol_LMA.u[end][idx_m]
    stds_lma[j] = sqrt(sol_LMA.u[end][idx_m2] - means_lma[j] ^ 2)
end

## Compute moments for the full model
@showprogress for (j, σ_b) in enumerate(σ_bs)
    traj = simulate_traj(rsi; tmax=tmax, u0=[1,0,0], ps=[σ_b])
    
    weights = diff(traj.times) / tmax
    μ_gp = sum(map(x -> x[2] * x[1], traj.states[1:end-1]) .* weights)
    μ_g = sum(map(x -> x[1], traj.states[1:end-1]) .* weights)
    μ_p = sum(map(x -> x[2], traj.states[1:end-1]) .* weights)
    μ_p2 = sum(map(x -> x[2] * x[2], traj.states[1:end-1]) .* weights)
    
    means_full[j] = μ_p
    stds_full[j] = sqrt(μ_p2 - μ_p^2)
    
    # While we're at it: estimate the effective binding rate for the reduction 
    ps_red[j] = σ_b * μ_gp / μ_g
end

## Compute moments for the reduced model
@showprogress for (j, σ_b_red) in enumerate(ps_red)
    rsi_p = ReactionSystemInfo(rsi_red; ps=[σ_b_red])
    traj_red = simulate_traj(rsi_p; tmax=tmax, u0=[1,0,0])
    
    weights = diff(traj_red.times) / tmax
    μ_p_red = sum(map(x -> x[2], traj_red.states[1:end-1]) .* weights)
    μ_p2_red = sum(map(x -> x[2] * x[2], traj_red.states[1:end-1]) .* weights)
    
    means_red[j] = μ_p_red
    stds_red[j] = sqrt(μ_p2_red - μ_p_red^2)
end

@save "moments.jld2" σ_bs ps_red means_full stds_full means_lma stds_lma means_red stds_red