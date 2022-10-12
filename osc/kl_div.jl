using ProgressMeter
using JLD2

include("cme.jl")
include("likelihoods.jl")
include("optim.jl")
include("hiddenlikelihoods_split.jl")

include("oscillator/osc.jl")

experiment = 2
@load "oscillator/system_$(experiment).jld2" gt_params params_noE params_1Gp params_noE1Gp params_1G params_noG;

##

# obsmat_noG = [ 1 0 0 0 0
#                0 1 0 0 1
#                0 0 1 0 0
#                0 0 0 1 0 ]

# hmat_noG = [ 0 0 0 0 1 ]

# u0_hfsp_noG = zeros(3)
# u0_hfsp_noG[1] = 1.0

# rsi_noG = ReactionSystemInfo(sys_noG; ps=params_noG)
# hsys_noG = HiddenFSPSystem(sys_cpt_noG, NaiveIndexHandler(), obsmat_noG, hmat_noG);
# hfunc_noG = ODEFunction{true}(build_rhs_split(hsys_noG))


# ##

# obsmat_1G = [ 1 0 0 0 0 0 1
#               0 1 0 0 0 0 0
#               0 0 1 0 0 0 1
#               0 0 0 1 0 0 0
#               0 0 0 0 1 0 0
#               0 0 0 0 0 1 0]

# hmat_1G = [ 0 0 0 0 0 0 1 ]

# u0_hfsp_1G = zeros(2)
# u0_hfsp_1G[1] = 1.0

# rsi_1G = ReactionSystemInfo(sys_1G; ps=params_1G)
# hsys_1G = HiddenFSPSystem(sys, NaiveIndexHandler(), obsmat_1G, hmat_1G);
# hfunc_1G = ODEFunction{true}(build_rhs_split(hsys_1G))

# ##

# obsmat_noE = [ 1 0 0 0 0 0
#                0 1 0 0 0 0
#                0 0 1 1 0 0
#                0 0 0 0 1 0
#                0 0 0 0 0 1 ]

# hmat_noE = [ 0 0 0 1 0 0 ]

# u0_hfsp_noE = zeros(u0[4]+1)
# u0_hfsp_noE[1] = 1.0

# rsi_noE = ReactionSystemInfo(sys_noE; ps=params_noE)
# hsys_noE = HiddenFSPSystem(sys_cpt_noE, NaiveIndexHandler(), obsmat_noE, hmat_noE);
# hfunc_noE = ODEFunction{true}(build_rhs_split(hsys_noE))

# ##

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

##

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

##

N = 25
tmax = 50000#if experiment == 3 5e6 else 2e5 end

#dkls_noG = zeros(N)
#dkls_1G = zeros(N)
#dkls_noE = zeros(N)
#dkls_1Gp = zeros(N)
#dkls_noE1Gp = zeros(N)

@load "oscillator/kl_$(experiment).jld2" dkls_noG dkls_1G dkls_noE dkls_1Gp dkls_noE1Gp

lk = Threads.SpinLock()

function save_progress()
    Threads.lock(lk)
    try
        @save "oscillator/kl_$(experiment).jld2" dkls_noG dkls_1G dkls_noE dkls_1Gp dkls_noE1Gp
    catch
        println("Cannot save...")
        @show dkls_noG dkls_1G dkls_noE dkls_1Gp dkls_noE1Gp
    finally
        Threads.unlock(lk)
    end
end

Threads.@threads for i in 1:N
    @time traj = simulate_traj(sys; tmax=tmax, u0=u0, ps=gt_params);

#     traj_noG = marg_traj_noG(traj)

#     @time lls_noG, u = solve_hfsp_split(hsys_noG, hfunc_noG, traj_noG, u0_hfsp_noG, gt_params;
#                                     solver=KenCarp4())

#     lps_noG = [ logpdf(rsi_noG, traj_noG, i) for i in 1:length(rsi_noG.ratefuncs) ]
#     dkl_noG = (sum(lls_noG) - sum(lps_noG)) / tmax
#     dkls_noG[i] = dkl_noG

#     save_progress()

#     traj_1G = marg_traj_1G(traj)

#     @time lls_1G, u = solve_hfsp_split(hsys_1G, hfunc_1G, traj_1G, u0_hfsp_1G, gt_params;
#                                        solver=KenCarp4())

#     lps_1G = [ logpdf(rsi_1G, traj_1G, i) for i in 1:length(rsi_1G.ratefuncs) ]
#     dkl_1G = (sum(lls_1G) - sum(lps_1G)) / tmax
#     dkls_1G[i] = dkl_1G

#     save_progress()
        
#     traj_noE = marg_traj_noE(traj)

#     @time lls_noE, u = solve_hfsp_split(hsys_noE, hfunc_noE, traj_noE, u0_hfsp_noE, [gt_params; 10; 0];
#                                        solver=KenCarp4())

#     lqs_noE = [ logpdf(rsi_noE, traj_noE, i) for i in 1:length(rsi_noE.ratefuncs) ]
#     dkl_noE = (sum(lls_noE) - sum(lqs_noE)) / tmax
#     dkls_noE[i] = dkl_noE

#     save_progress()

#     traj_1Gp = marg_traj_1Gp(traj)
#     @time lls_1Gp, u = solve_hfsp_split(hsys_1Gp, hfunc_1Gp, traj_1Gp, u0_hfsp_1Gp, gt_params;
#                                        solver=KenCarp4())

#     lps_1Gp = [ logpdf(rsi_1Gp, traj_1Gp, i) for i in 1:length(rsi_1Gp.ratefuncs) ]
#     dkl_1Gp = (sum(lls_1Gp) - sum(lps_1Gp)) / tmax
#     dkls_1Gp[i] = dkl_1Gp

#     save_progress()

    traj_noE1Gp = marg_traj_noE1Gp(traj)
    @time lls_noE1Gp, u = solve_hfsp_split(hsys_noE1Gp, hfunc_noE1Gp, traj_noE1Gp, u0_hfsp_noE1Gp, [gt_params; 10; 0];
                                       solver=KenCarp4(), atol=1e-6, rtol=1e-6)

    lps_noE1Gp = [ logpdf(rsi_noE1Gp, traj_noE1Gp, i) for i in 1:length(rsi_noE1Gp.ratefuncs) ]
    dkl_noE1Gp = (sum(lls_noE1Gp) - sum(lps_noE1Gp)) / tmax
    dkls_noE1Gp[i] = dkl_noE1Gp

    #save_progress()
end