using JLD2
using OrdinaryDiffEq
using LinearAlgebra

include("../cme.jl")
include("osc.jl")

## LNA code

using Symbolics
using MatrixEquations

# Generate symbolic drift and diffusion matrices
function assemble_LNA_drift_diff(sys; combinatoric_ratelaw=true)
    states = Catalyst.get_states(sys)
    eqs = Catalyst.get_eqs(sys)
    fs = [ Catalyst.oderatelaw(rx; combinatoric_ratelaw=combinatoric_ratelaw) for rx in eqs ]
    S = Catalyst.netstoichmat(sys)
    
    J = [ sum(S[i,r] * Symbolics.derivative(fs[r], states[j]) for r in 1:length(eqs)) for i in 1:length(states), j in 1:length(states) ]
    D = [ Num(sum(S[i,r] * S[j,r] * fs[r] for r in 1:length(eqs))) for i in 1:length(states), j in 1:length(states) ]
    
    J, D
end

function compute_LNA_ss_mean(sys, u0, ps; solver=KenCarp4())
    ssprob = ODEProblem(sys, u0, (0, 1e6), ps)
    sol = solve(ssprob, solver)
    sol.u[end]
end
    
# Compute steady-state drift and diffusion matrices
function compute_LNA_drift_diff_ss(sys, u0, ps; kwargs...)
    ss_mean = compute_LNA_ss_mean(sys, u0, ps; kwargs...)
    
    J, D = assemble_LNA_drift_diff(sys)
    
    subs_dict_vals = Dict(sys.states[i] => ss_mean[i] for i in 1:length(sys.states))
    subs_dict_ps = Dict(sys.ps[i] => ps[i] for i in 1:length(sys.ps))
    subs_dict = union(subs_dict_vals, subs_dict_ps)
    
    J_ev = map(x -> Catalyst.value(substitute(x, subs_dict)), J)
    D_ev = map(x -> Catalyst.value(substitute(x, subs_dict)), D)
    
    J_ev, D_ev
end
    
# Compute steady-state covariance  using the Lyapunov Equation
function compute_LNA_ss_cov(sys, u0, ps; kwargs...)
    J, D = compute_LNA_drift_diff_ss(sys, u0, ps; kwargs...)
    lyapc(Float64.(J), Float64.(D))
end

function LNA_powerspec(J, D, ω)
    A = J + I(size(J, 1)) .* (ω * im)
    ret = A \ D
    ret = ret'
    ret = A \ ret
    1 / (2 * pi) * ret
end

## Apply the LNA to all systems

@load "params.jld2" gt_params params_noE params_1Gp params_noE1Gp params_1G params_noG

J_full, D_full = compute_LNA_drift_diff_ss(sys, u0, gt_params)
J_noE, D_noE = compute_LNA_drift_diff_ss(sys_noE, u0_noE, params_noE)
J_1Gp, D_1Gp = compute_LNA_drift_diff_ss(sys_1Gp, u0_1Gp, params_1Gp)
J_noE1Gp, D_noE1Gp = compute_LNA_drift_diff_ss(sys_noE1Gp, u0_noE1Gp, params_noE1Gp)
J_1G, D_1G = compute_LNA_drift_diff_ss(sys_1G, u0_1G, params_1G)
J_noG, D_noG = compute_LNA_drift_diff_ss(sys_noG, u0_noG, params_noG)

## Compute mRNA and Protein power spectra

ωs = 10 .^ (-4:0.01:-1)

P_full_mRNA = [ real(diag(LNA_powerspec(J_full, D_full, ω))[2]) for ω in ωs ] ./ 1e6
P_noE_mRNA = [ real(diag(LNA_powerspec(J_noE, D_noE, ω))[2]) for ω in ωs ] ./ 1e6
P_1Gp_mRNA = [ real(diag(LNA_powerspec(J_1Gp, D_1Gp, ω))[2]) for ω in ωs ] ./ 1e6
P_noE1Gp_mRNA = [ real(diag(LNA_powerspec(J_noE1Gp, D_noE1Gp, ω))[2]) for ω in ωs ] ./ 1e6
P_1G_mRNA = [ real(diag(LNA_powerspec(J_1G, D_1G, ω))[2]) for ω in ωs ] ./ 1e6
P_noG_mRNA = [ real(diag(LNA_powerspec(J_noG, D_noG, ω))[1]) for ω in ωs ] ./ 1e6

P_full_prot = [ real(diag(LNA_powerspec(J_full, D_full, ω))[3]) for ω in ωs ] ./ 1e6
P_noE_prot = [ real(diag(LNA_powerspec(J_noE, D_noE, ω))[3]) for ω in ωs ] ./ 1e6
P_1Gp_prot = [ real(diag(LNA_powerspec(J_1Gp, D_1Gp, ω))[3]) for ω in ωs ] ./ 1e6
P_noE1Gp_prot = [ real(diag(LNA_powerspec(J_noE1Gp, D_noE1Gp, ω))[3]) for ω in ωs ] ./ 1e6
P_1G_prot = [ real(diag(LNA_powerspec(J_1G, D_1G, ω))[3]) for ω in ωs ] ./ 1e6
P_noG_prot = [ real(diag(LNA_powerspec(J_noG, D_noG, ω))[2]) for ω in ωs ] ./ 1e6

@save "powerspectrum.jl" ωs P_full_mRNA P_noE_mRNA P_1Gp_mRNA P_noE1Gp_mRNA P_1G_mRNA P_noG_mRNA P_full_prot P_noE_prot P_1Gp_prot P_noE1Gp_prot P_1G_prot P_noG_prot