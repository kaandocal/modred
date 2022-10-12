# Estimate KL divergences for the genetic oscillator via Monte Carlo
using ProgressMeter
using JLD2

include("../cme.jl")
include("../likelihoods.jl")
include("../hiddenlikelihoods.jl")

include("osc.jl")

@load "params.jld2" gt_params params_noE params_1Gp params_noE1Gp params_1G params_noG

## Set up the relevant ODE systems to compute marginal likelihoods of a trajectory

# Model I
# The observation matrix defines the map (full species) -> (reduced species)
# The species are given by the order in the relevant reaction systems in osc.jl
obsmat_noE = [ 1 0 0 0 0 0
               0 1 0 0 0 0
               0 0 1 1 0 0
               0 0 0 0 1 0
               0 0 0 0 0 1 ]

# The hidden matrix defines the complementary map (full species) -> (hidden species)
# The joint map (full species) -> (reduced species; hidden_species) has to be invertible
# ie. the state of the full system is uniquely defined by the observed species and 
# the hidden species
hmat_noE = [ 0 0 0 1 0 0 ]

# Initial values for the hidden species (for Model I, this is the number of bound enzymes)
u0_hfsp_noE = zeros(u0[4]+1)
u0_hfsp_noE[1] = 1.0

rsi_noE = ReactionSystemInfo(sys_noE; ps=params_noE)
hsys_noE = HiddenFSPSystem(sys_cpt_noE, NaiveIndexHandler(), obsmat_noE, hmat_noE);
hfunc_noE = ODEFunction{true}(build_rhs_split(hsys_noE))

# Model II
obsmat_1Gp = [ 1 0 0 0 0 0 0
               0 1 0 0 0 0 0
               0 0 1 0 0 1 0
               0 0 0 1 0 0 0
               0 0 0 0 1 0 0
               0 0 0 0 0 1 1 ]

hmat_1Gp = [ 0 0 0 0 0 0 1 ]

u0_hfsp_1Gp = zeros(2)
u0_hfsp_1Gp[1:2] .= 0.5

rsi_1Gp = ReactionSystemInfo(sys_1Gp; ps=params_1Gp)
hsys_1Gp = HiddenFSPSystem(sys, NaiveIndexHandler(), obsmat_1Gp, hmat_1Gp);
hfunc_1Gp = ODEFunction{true}(build_rhs_split(hsys_1Gp))

# Model I + II
obsmat_noE1Gp = [ 1 0 0 0 0 0
                  0 1 0 0 0 0
                  0 0 1 1 1 0
                  0 0 0 0 1 1 ]

hmat_noE1Gp = [ 0 0 0 0 0 1
                0 0 0 1 0 0 ]

u0_hfsp_noE1Gp = zeros(2, u0[4]+1)
u0_hfsp_noE1Gp[1,1] = 1.0

rsi_noE1Gp = ReactionSystemInfo(sys_noE1Gp; ps=params_noE1Gp)
hsys_noE1Gp = HiddenFSPSystem(sys_cpt_noE1Gp, NaiveIndexHandler(), obsmat_noE1Gp, hmat_noE1Gp);
hfunc_noE1Gp = ODEFunction{true}(build_rhs_split(hsys_noE1Gp))

# Model IV
obsmat_1G = [ 1 0 0 0 0 0 1
              0 1 0 0 0 0 0
              0 0 1 0 0 0 1
              0 0 0 1 0 0 0
              0 0 0 0 1 0 0
              0 0 0 0 0 1 0]

hmat_1G = [ 0 0 0 0 0 0 1 ]

u0_hfsp_1G = zeros(2)
u0_hfsp_1G[1] = 1.0

rsi_1G = ReactionSystemInfo(sys_1G; ps=params_1G)
hsys_1G = HiddenFSPSystem(sys, NaiveIndexHandler(), obsmat_1G, hmat_1G);
hfunc_1G = ODEFunction{true}(build_rhs_split(hsys_1G))

# Model V
obsmat_noG = [ 1 0 0 0 0
               0 1 0 0 1
               0 0 1 0 0
               0 0 0 1 0 ]

hmat_noG = [ 0 0 0 0 1 ]

u0_hfsp_noG = zeros(3)
u0_hfsp_noG[1] = 1.0

rsi_noG = ReactionSystemInfo(sys_noG; ps=params_noG)
hsys_noG = HiddenFSPSystem(sys_cpt_noG, NaiveIndexHandler(), obsmat_noG, hmat_noG);
hfunc_noG = ODEFunction{true}(build_rhs_split(hsys_noG))

## Estimate KL divergences

# Simulate one long reference trajectory
tmax = 1e6
traj = simulate_traj(sys; tmax=tmax, u0=u0, ps=gt_params);

# Model I
traj_noE = marg_traj_noE(traj)
ll_noE, u = solve_hfsp(hsys_noE, hfunc_noE, traj_noE, u0_hfsp_noE, [gt_params; 10; 0];
                       solver=KenCarp4())

dkl_noE = (ll_noE[] - logpdf(rsi_noE, traj_noE)) / tmax

# Model II
traj_1Gp = marg_traj_1Gp(traj)
ll_1Gp, u = solve_hfsp(hsys_1Gp, hfunc_1Gp, traj_1Gp, u0_hfsp_1Gp, gt_params;
                       solver=KenCarp4())

dkl_1Gp = (ll_1Gp[] - logpdf(rsi_1Gp, traj_1Gp)) / tmax

# Model I + II
traj_noE1Gp = marg_traj_noE1Gp(traj)
lls_noE1Gp, u = solve_hfsp(hsys_noE1Gp, hfunc_noE1Gp, traj_noE1Gp, u0_hfsp_noE1Gp, [gt_params; 10; 0];
                           solver=KenCarp4(), atol=1e-6, rtol=1e-6)

dkl_noE1Gp = (ll_noE1Gp[] - logpdf(rsi_noE1Gp, traj_noE1Gp)) / tmax

# Model IV
traj_1G = marg_traj_1G(traj)
ll_1G, u = solve_hfsp(hsys_1G, hfunc_1G, traj_1G, u0_hfsp_1G, gt_params;
                      solver=KenCarp4())

dkl_1G = (ll_1G[] - logpdf(rsi_1G, traj_1G)) / tmax

# Model V
traj_noG = marg_traj_noG(traj)
ll_noG, u = solve_hfsp(hsys_noG, hfunc_noG, traj_noG, u0_hfsp_noG, gt_params;
                       solver=KenCarp4())

dkl_noG = (ll_noG[] - logpdf(rsi_noG, traj_noG)) / tmax

@save "dkls.jld2" dkls_noG dkls_1G dkls_noE dkls_1Gp dkls_noE1Gp
