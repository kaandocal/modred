# Compute KL divergence rate between telegraph model and its constitutive 
# approximation as shown in Fig. 3d
using Catalyst
using FiniteStateProjection
using DifferentialEquations
using ProgressMeter 
using JLD2

include("../likelihoods.jl")
include("../hiddenlikelihoods.jl")

rn = @reaction_network begin
    σ_on * (1 - G), 0 --> G
    σ_off, G --> 0
    ρ, G --> G + P
end σ_on σ_off ρ

rn_red = @reaction_network begin
    ρ, 0 --> P 
end ρ

u0 = [ 0., 0. ] 

hsys = HiddenFSPSystem(rn, DefaultIndexHandler(), [ 0 1 ], [ 1 0 ])
func = ODEFunction{true}(build_rhs(hsys))

cc = 10 .^ (-2:0.1:2)
pons = [ 0.1, 0.3, 0.5, 0.7, 0.9 ]
N = 100
dkls = zeros(length(pons), length(cc), N)


# We estimate the KL divergence rate in two ways: by computing the difference in 
# log-likelihoods for a single, long trajectory
for (j, pon) in enumerate(pons)
    @showprogress for (i, c) in enumerate(cc)
        params = [ pon * c, (1 - pon) * c, 1 ]
        red_params = [ pon ]

        rsi_red = ReactionSystemInfo(rn_red; ps=red_params);

        T = 10000.
        Threads.@threads for n in 1:N
            traj = simulate_traj(rn; u0=[0,0], tmax=T, ps=params)
            traj_red = marginalise(traj, (2,))
            
            # The likelihood under the reduced model can be computed directly
            ll_p = logpdf(rsi_red, traj_red)

            # We need to do filtering on the gene state to compute the marginal
            # likelihood under the full model
            ls, u = solve_hfsp(hsys, func, traj_red, [1., 0.], params)
            ll_q = ls[]
            
            dkls[j,i,n] = ll_q - ll_p
        end
        
        @save "dkls.jld2" cc pons dkls
    end
end
