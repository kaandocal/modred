# Compute KL divergence for the Michaelis-Menten model and its QSSA and QEA
# versions
include("../cme.jl")
include("../likelihoods.jl")
include("../hiddenlikelihoods.jl")
include("../viz.jl")

include("mm.jl")

using ProgressMeter
using JLD2

func = ODEFunction{true}(build_rhs(sys))

cc = 10. .^ (-3:0.1:0)
EE = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 20, 25, 31, 40, 52, 65, 80, 100 ]

dkls_qea = zeros(length(cc), length(EE))
dkls_qssa = zeros(length(cc), length(EE))

N = 1000
T = 10000.

for (i, c) in enumerate(cc)
    @showprogress for (j, E0) in enumerate(EE)
        # Apply rescaling using c and E0
        params = copy(gt_params)
        params[2] *= c
        params[3] *= c

        params[4] = E0
        params[5] = 100

        ϵ = E0 / 10
        params[1] /= ϵ
        params[2] /= ϵ
        params[3] /= ϵ

        rsi_qea = ReactionSystemInfo(rs_qea; ps=params);
        rsi_qssa = ReactionSystemInfo(rs_qssa; ps=params);

        # Project trajectory
        trajs = simulate_trajs(rs, N; u0=[E0, 0], tmax=T, ps=params)
        trajs_red = map(traj -> clean_traj(marginalise(traj, (2,))), trajs)
        
        u0 = zeros(E0+1)
        u0[E0+1] = 1.0
        
        Threads.@threads for n in 1:N
            traj = trajs[n]
            traj_red = trajs_red[n]
            
            # Compute log-likelihood of marginal trajectory using full model
            ls, u = solve_hfsp(sys, func, traj_red, u0, params;
                               solver=KenCarp4(), abstol=1e-6, reltol=1e-3)
            
            dkls_qea[i,j] += (ls[] - logpdf(rsi_qea, traj_red)) / N
            dkls_qssa[i,j] += (ls[] - logpdf(rsi_qssa, traj_red)) / N
        end
    end
    
    @save "dkls.jld2" cc EE dkls_qea dkls_qssa
end