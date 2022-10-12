include("./cme.jl")

using AbstractAlgebra: matrix, snf_with_transform, ZZ
using RuntimeGeneratedFunctions
using OrdinaryDiffEq
RuntimeGeneratedFunctions.init(@__MODULE__)
using MacroTools
using RecursiveArrayTools

using FiniteStateProjection
using FiniteStateProjection: AbstractIndexHandler, singleindices, pairedindices

struct HiddenFSPSystem{IHT <: AbstractIndexHandler, RT1, RT2}
    rs::ReactionSystem
    ih::IHT
    nobs::Int
    obsmat_aug::Matrix{Int}
    iobsmat_aug::Matrix{Rational{Int}}
    netstoichmat::Matrix{Int}
    hidden_rfs::RT1
    visible_rfs::RT2
end

# function HiddenFSPSystem(rs::ReactionSystem, ih::AbstractIndexHandler, obsmat::AbstractMatrix{Int};
#                          combinatoric_ratelaw::Bool=true)
#     obsmat_conv = matrix(ZZ, obsmat)
#     S, T, U = snf_with_transform(obsmat_conv)
    
#     n = findfirst(i -> all(S[:,i] .== 0), 1:size(obsmat_conv, 2))
    
#     if n === nothing 
#         n = size(obsmat_conv, 2) + 1
#     end
    
#     C = Int.(U[:,n:size(obsmat_conv, 2)]')
    
#     for i in 1:size(C, 1)
#         if all(C[i,:] .<= 0) 
#             C[i,:] .= -C[i,:]
#         end
#     end
    
#     HiddenFSPSystem(rs, ih, obsmat, C; combinatoric_ratelaw)
# end


function HiddenFSPSystem(rs::ReactionSystem, ih::AbstractIndexHandler, obsmat::AbstractMatrix{Int}, hidden_mat::AbstractMatrix{Int};
                         combinatoric_ratelaw::Bool=true)
    
    obsmat_conv = matrix(ZZ, obsmat)
    S, T, U = snf_with_transform(obsmat_conv)
    
    nobs = size(obsmat, 1)
    
    n = findfirst(i -> all(S[:,i] .== 0), 1:size(obsmat_conv, 2))
    
    obsmat_aug = [ obsmat; hidden_mat ]
    iobsmat_aug = inv(Rational.(obsmat_aug))
    
    So = Catalyst.netstoichmat(rs)' * obsmat_aug'
    
    hid_eq_idcs = filter(i -> all(So[i,1:n-1] .== 0), 1:size(So, 1))
    vis_eq_idcs = filter(i -> !all(So[i,1:n-1] .== 0), 1:size(So, 1))
    
    rfs = FiniteStateProjection.create_ratefuncs(rs, FiniteStateProjection.NaiveIndexHandler(0); 
                                                 combinatoric_ratelaw=combinatoric_ratelaw)
    hidden_rfs = rfs[hid_eq_idcs]
    visible_rfs = rfs[vis_eq_idcs]
    
    nsm_raw = Catalyst.netstoichmat(rs)
    nsm = similar(nsm_raw)
    for i in 1:length(hidden_rfs)
        nsm[:,i] .= nsm_raw[:,hid_eq_idcs[i]]
    end
    
    for i in 1:length(visible_rfs)
        nsm[:,length(hidden_rfs) + i] .= nsm_raw[:,vis_eq_idcs[i]]
    end
    
    HiddenFSPSystem(rs, ih, nobs, obsmat_aug, iobsmat_aug, nsm, hidden_rfs, visible_rfs)
end

Catalyst.netstoichmat(sys::HiddenFSPSystem) = sys.netstoichmat

conv_netstoichmat(sys::HiddenFSPSystem) = sys.obsmat_aug * Catalyst.netstoichmat(sys)
hidden_netstoichmat(sys::HiddenFSPSystem) = conv_netstoichmat(sys)[sys.nobs+1:end,:]
visible_netstoichmat(sys::HiddenFSPSystem) = conv_netstoichmat(sys)[1:sys.nobs,length(sys.hidden_rfs)+1:end]

function convert_cartidx(sys::HiddenFSPSystem, state_hid, state_obs)
    sys.iobsmat_aug * [ state_obs; (Tuple(state_hid) .- sys.ih.offset)... ]
end

## FSP-related stuff

wrap(u::AbstractArray) = ArrayPartition([log(sum(u))], u)
unwrap(u_wrapped::ArrayPartition) = (u_wrapped.x[1], u_wrapped.x[2])

function unpackparams(sys::HiddenFSPSystem, psym::Symbol)::Expr
    param_names = Expr(:tuple, map(par -> par.name, Catalyst.params(sys.rs))...)
     
    quote 
        $(param_names) = $(psym)
    end
end

function build_rhs_header(sys::HiddenFSPSystem)::Expr
    quote 
        dls, du = $(unwrap)(du_wrapped)
        ls, u = $(unwrap)(u_wrapped)
        sys = $(sys)
        ps, state_obs = p
        $(unpackparams(sys, :ps))
    end
end

function build_rhs_firstpass(sys::HiddenFSPSystem)    
    rfs = tuple(sys.hidden_rfs..., sys.visible_rfs...) 
        
    first_line = :(du[idx_in_hid] = -u[idx_in_hid] * max(0,$(rfs[1].body)))
    other_lines = (:(du[idx_in_hid] -= u[idx_in_hid] * max(0, $(rf.body))) for rf in rfs[2:end])
    
    quote
        for idx_in_hid in singleindices(sys.ih, u)
            idx_in = convert_cartidx(sys, idx_in_hid, state_obs)
            $first_line
            $(other_lines...)
        end
    end
end

function build_rhs_secondpass(sys::HiddenFSPSystem)::Expr
    S = hidden_netstoichmat(sys)'
    ret = Expr(:block)
    
    for (i, rf) in enumerate(sys.hidden_rfs)
        ex = quote
            for (idx_in_hid, idx_out_hid) in pairedindices(sys.ih, u, $(CartesianIndex(S[i,:]...)))
                idx_in = convert_cartidx(sys, idx_in_hid, state_obs)
                du[idx_out_hid] += u[idx_in_hid] * max(0, $(rf.body))
            end
        end
        
        append!(ret.args, ex.args)
    end
    
    return ret
end

function build_rhs_ex(sys::HiddenFSPSystem; striplines::Bool=true) 
    sl = striplines ? MacroTools.striplines : identity
    
    header = build_rhs_header(sys) |> sl
    first_pass = build_rhs_firstpass(sys) |> sl
    second_pass = build_rhs_secondpass(sys) |> sl
    footer = :( dls[] = sum(du); du .-= u .* dls[]; ) |> sl
    
    body = Expr(:block, header, first_pass, second_pass, footer)
    
    ex = :((du_wrapped, u_wrapped, p, t) -> $(body)) 
    
    ex = ex |> MacroTools.flatten
    
    ex
end

function build_rhs(sys::HiddenFSPSystem) 
    @RuntimeGeneratedFunction(build_rhs_ex(sys; striplines=false))
end

## Likelihood stuff

function process_transition!(u_wrapped, sys::HiddenFSPSystem, ps, curr::T, next::T, t) where {T}
    vnsm = visible_netstoichmat(sys)
    S = hidden_netstoichmat(sys)
    reac_idcs = filter(i -> vnsm[:,i] == (next .- curr), 1:size(vnsm, 2))
    
    s, u = unwrap(u_wrapped)
    
    @assert all(u .>= 0)
    
    u_old = copy(u)
    fill!(u, 0)
    
    for reac_idx in reac_idcs
        reacfunc = sys.visible_rfs[reac_idx]
        for (idx_in_hid, idx_out_hid) in pairedindices(sys.ih, u_old, CartesianIndex(S[:,reac_idx+length(sys.hidden_rfs)]...))
            state_full = convert_cartidx(sys, idx_in_hid, curr)

            rate = reacfunc(state_full, t, ps...)
            
            u[idx_out_hid] += rate * u_old[idx_in_hid]
        end
    end
    
    @assert all(u .>= 0)
    su = sum(u)
    s[] += log(su)
    
    if !iszero(su)
        u ./= su
    end
end


function solve_hfsp(sys::HiddenFSPSystem, func, traj::Trajectory, u0, ps;
                    solver=KenCarp4(), solverkwargs...) where {T}
    u_wrapped = wrap(u0)
    
    dts = diff(traj.times)
    n_species = length(first(traj.states))
    
    curr_t = 0.0
    
    prob = ODEProblem(func, u_wrapped, (0., traj.times[1]), (ps, traj.states[1]))#, save_everystep=false)
    
    ll = 0
    
    for i in range(1, stop=length(dts)-1)
        dt = dts[i]
        curr = traj.states[i]
        next = traj.states[i+1]
        
        if next == curr
            curr_t += dt
            continue
        end
        
        prob = remake(prob, u0=u_wrapped, tspan=(traj.times[i+1] - curr_t - dt, traj.times[i+1]), 
                      p=(ps, traj.states[i]))
        sol = solve(prob, solver; solverkwargs...)
        
        if sol.retcode != :Success
            return unwrap(u_wrapped)
        end
        
        u_wrapped.x[1] .= sol.u[end].x[1]
        u_wrapped.x[2] .= abs.(sol[end].x[2])
        
        isfinite(u_wrapped.x[1][]) || return unwrap(u_wrapped)
        
        # Handle jumps
        process_transition!(u_wrapped, sys, ps, curr, next, traj.times[i+1])        
        isfinite(u_wrapped.x[1][]) ||  return unwrap(u_wrapped)
        
        ll += u_wrapped.x[1][]
        u_wrapped.x[1] .= 0
        
        curr_t = 0.
    end
    
    if traj.states[end-1] != traj.states[end]
        @warn "trajectory seems truncated"
    else
        prob = remake(prob, u0=u_wrapped, tspan=(traj.times[end] - curr_t, traj.times[end]), 
                      p=(ps, traj.states[end-1]))
        sol = solve(prob, solver; solverkwargs...)
    
        u_wrapped.x[1] .= sol.u[end].x[1]
        u_wrapped.x[2] .= abs.(sol[end].x[2])
    end
    
    u_wrapped.x[1][] += ll
    
    return unwrap(u_wrapped)
end
