# Compute KL divergence rates between full and reduced AFL as shown in Fig. 4d
include("../cme.jl")
include("../likelihoods.jl")

include("afl.jl")

using ProgressMeter
using LogExpFunctions
using JLD2

# Binding rates for which we want to compute the KL divergence
σ_bs = [ 0.1, 0.2, 0.4, 0.7, 0.9, 1.3, 1.5, (2:0.1:3.5)..., 4.0, 4.5, 5, 6, 8, 10 ]
tmax = 1e5

# We estimate the KL divergence rate in two ways: by computing the difference in 
# log-likelihoods for a single, long trajectory and analytically by using Eq. 39 
# in the paper, where we still use Monte Carlo sampling to estimate the relevant
# moments
ps_red = zeros(length(σ_bs))
dkls_ssa = similar(ps_red)
dkls_an = similar(dkls_ssa)

p = Progress(length(σ_bs))
Threads.@threads for j in 1:length(σ_bs)
    σ_b = σ_bs[j]
    traj = simulate_traj(rsi; tmax=tmax, u0=[1,0,0], ps=[σ_b])
    weights = diff(traj.times) / tmax
    
    # Compute E[P | G = 1] to estimate reduced parameter
    μ_gp = sum(map(x -> x[2] * x[1], traj.states[1:end-1]) .* weights)
    μ_g = sum(map(x -> x[1], traj.states[1:end-1]) .* weights)
    ps_red[j] = σ_b * μ_gp / μ_g
    
    # Approximate the KL divergence using the analytical formula in the paper
    μ_gplogp = sum(map(x -> x[1] * xlogx(x[2]), traj.states[1:end-1]) .* weights)
    dkls_an[j] = σ_b * μ_g * (μ_gplogp / μ_g - μ_gp / μ_g * log(μ_gp / μ_g)) 
    
    # Compute log-likelihoods of the trajectory for the original & reduced models
    rsi_q = ReactionSystemInfo(rsi; ps=[σ_b])
    rsi_p = ReactionSystemInfo(rsi_red; ps=[ps_red[j]])
    
    dkls_ssa[j] = logpdf(rsi_q, traj) - logpdf(rsi_p, traj)
    
    next!(p)
end

@save "dkls.jld2" σ_bs dkls_ssa dkls_an