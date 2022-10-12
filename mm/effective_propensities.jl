# Compute effective propensities for the reduced Michaelis-Menten model 
# as shown in 5d, following Eq. (13) and using Monte-Carlo integration
include("../cme.jl")
include("../likelihoods.jl")
include("../hiddenlikelihoods.jl")
include("../viz.jl")

include("mm.jl")

using ProgressMeter
using JLD2

cc = 10. .^ (-3:0.5:1)
EE = [1, 5, 10, 20, 35 ]
N = 5000
T = 10000.

props_qea = zeros(length(cc), S0+1)
props_qssa = zeros(length(EE), S0+1)

# Compute effective propensities for the QEA rescaling
@showprogress for (i, c) in enumerate(cc)
    # Apply rescaling
    params = copy(gt_params)
    params[2] *= c
    params[3] *= c
    
    # Histogram of occupation times 
    hst = zeros(S0+1, E0+1)
    
    trajs = simulate_trajs(rs, N; u0=[E0, 0], tmax=T, ps=params)
    for traj in trajs
        for i in 1:length(traj.states) - 1
            s = traj.states[i]
            hst[s[2]+1,s[1]+1] += traj.times[i+1] - traj.times[i]
        end
    end
    
    # The effective propensity is the mean propensity in the original model
    # conditioned on the present substrate molecules
    hst ./= sum(hst, dims=2)
    props_qea[i,:] .= gt_params[1] .* (E0 .- sum(hst' .* (0:E0), dims=1)[:])
end

# Compute effective propensities for the QSSA rescaling
@showprogress for (i, E0) in enumerate(EE)
    # Apply rescaling
    e_resc = E0 / 10
    params = copy(gt_params)
    params[4] *= e_resc
    params[1] /= e_resc
    params[2] /= e_resc
    params[3] /= e_resc
    
    # Histogram of occupation times
    hst = zeros(S0+1, Int(E0)+1)
    
    trajs = simulate_trajs(rs, N; u0=[E0, 0], tmax=T, ps=params)
    for traj in trajs
        for i in 1:length(traj.states) - 1
            s = traj.states[i]
            hst[s[2]+1,s[1]+1] += traj.times[i+1] - traj.times[i]
        end
    end
    
    Km = (params[1] + params[3]) / params[2]
    vmax = params[1] * params[4]
    
    # The effective propensity is the mean propensity in the original model
    # conditioned on the present substrate molecules
    hst ./= sum(hst, dims=2)
    
    props_qssa[i,:] .= gt_params[1] .* (EE[i] .- sum(hst' .* (0:E0), dims=1)[:])
end

@save "eff_props.jld2" cc props_qea props_qssa