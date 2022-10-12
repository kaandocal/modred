using ProgressMeter
using JLD2
using Optim

include("../cme.jl")
include("../likelihoods.jl")

include("osc.jl")

params_noE = [ gt_params[1:3]; 0; gt_params[7:end] ]
params_1G = [ gt_params[1:end-2]; 0; 0; gt_params[end] ]
params_1Gp = [ gt_params[1:8]; 0; 0; 0 ]
params_noE1Gp = [ gt_params[1:3]; 0; gt_params[7:8]; 0; 0; 0 ]
params_noG = [ gt_params[1]; 0; 0; 0; gt_params[3:7] ]

opts = Optim.Options(g_tol = 1e-12, time_limit=300);

# Simulate one long reference trajectory (longer trajectory => better estimates)
tmax = 2e4#6
traj = simulate_traj(sys; tmax=tmax, u0=u0, ps=gt_params);

# Model I
rsi_noE = ReactionSystemInfo(sys_noE; ps=params_noE)
traj_noE = marg_traj_noE(traj);

loss_noE(p) = -logpdf(rsi_noE, traj_noE; ps=[ gt_params[1:3]; exp(p[]); gt_params[7:end]])

sol_noE = Optim.optimize(loss_noE, 5 .* randn(1), BFGS(), opts; autodiff=:forward)

params_noE = [ gt_params[1:3]; exp(sol_noE.minimizer[]); gt_params[7:end] ]

# Model II
rsi_1Gp = ReactionSystemInfo(sys_1Gp; ps=params_1Gp)
traj_1Gp = marg_traj_1Gp(traj);

loss_1Gp(log_ps) = -logpdf(rsi_1Gp, traj_1Gp, (Val(7), Val(8), Val(9)); ps=[gt_params[1:8]; exp.(log_ps)])

sol_1Gp = Optim.optimize(loss_1Gp, randn(3), BFGS(), opts; autodiff=:forward)
params_1Gp = [ gt_params[1:8]; exp.(sol_1Gp.minimizer) ]

# Model I + II
rsi_noE1Gp = ReactionSystemInfo(sys_noE1Gp; ps=params_noE1Gp)
traj_noE1Gp = marg_traj_noE1Gp(traj);

loss_noE1Gp(p) = -logpdf(rsi_noE1Gp, traj_noE1Gp; 
                         ps=[gt_params[1:3]; exp(p[1]); gt_params[7:8]; exp.(p[2:4])]) / tmax

sol_noE1Gp = Optim.optimize(loss_noE1Gp, randn(4), BFGS(); autodiff=:forward)
params_noE1Gp = [ gt_params[1:3]; exp(sol_noE1Gp.minimizer[1]); gt_params[7:8]; exp.(sol_noE1Gp.minimizer[2:4]) ]

# Model IV
rsi_1G = ReactionSystemInfo(sys_1G; ps=params_1G)
traj_1G = marg_traj_1G(traj);

loss_1G((log_v, log_Kd)) = -logpdf(rsi_1G, traj_1G, Val(8); ps=[gt_params[1:end-2]; exp(log_v); exp(log_Kd); gt_params[end]]) / tmax

sol_1G = Optim.optimize(loss_1G, 5 .* randn(2), Newton(), opts; autodiff=:forward)

params_1G = [gt_params[1:end-2]; exp.(sol_1G.minimizer); gt_params[end]]

# Model V
rsi_noG = ReactionSystemInfo(sys_noG; ps=params_noG)
traj_noG = marg_traj_noG(traj);

loss_noG(log_keff) = -logpdf(rsi_noG, traj_noG, Val(1); ps=[gt_params[1]; exp.(log_keff); gt_params[3:7]])
sol_noG = Optim.optimize(loss_noG, randn(3), NelderMead(), opts; autodiff=:forward)

params_noG .= [ gt_params[1]; exp.(sol_noG.minimizer); gt_params[3:7] ]

@save "params.jld2" gt_params params_noE params_1Gp params_noE1Gp params_1G params_noG