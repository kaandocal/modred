using Catalyst
using JumpProcesses
using StaticArrays
using SciMLBase
using SymbolicUtils
using ModelingToolkit
using StatsBase
const MT = ModelingToolkit

# Utilities

function speciesnames(rs::ReactionSystem)
    return map(spec -> SymbolicUtils.operation(spec).name, species(rs))
end

###################### Trajectories ######################

const Species = Int
const Marginal = Union{Species,Vector{Species}}

mutable struct Trajectory{State <: AbstractVector{Species}}
    times::Vector{Float64}
    states::Vector{State}
    n_species::Int64
end

extract_marg(vec::AbstractVector, marg::Species)::Int64 = vec[marg]
extract_marg(vec::AbstractVector, marg::AbstractVector{Species})::Int64 = sum(vec[spec] for spec in marg)
    
function marginalise(traj::Trajectory, margs::NTuple{N,Marginal})::Trajectory{SVector{N,Int64}} where {N}
    extract = vec -> SVector{N,Int64}(map(marg -> extract_marg(vec, marg), margs))
    return Trajectory{SVector{N,Int64}}(traj.times, map(extract, traj.states), N)
end

function truncate(traj::Trajectory{State}, tmax::Float64)::Trajectory{State} where State
    @assert traj.states[end] == traj.states[end-1]
    idx_trunc = findfirst(t -> t >= tmax, traj.times)
    if idx_trunc === nothing
        return traj
    end
    
    times_new = traj.times[1:idx_trunc]
    times_new[end] = times_new[end-1]
    
    states_new = traj.states[1:idx_trunc]
    states_new[end] = states_new[end-1]
    
    return Trajectory{State}(times_new, states_new, traj.n_species)    
end

function get_traj_states!(out, traj::Trajectory{State}, tt::AbstractVector{<:Real}) where State
    @assert traj.states[end] == traj.states[end-1]
    @assert all(diff(tt) .>= 0)
    @assert tt[1] > 0
    
    ti = 1
    for (idx, t) in enumerate(traj.times)
        if t < tt[ti]
            continue
        end
        
        while ti <= length(tt) && t > tt[ti]
            out[ti,:] .= traj.states[idx-1]
            ti += 1
        end
        
        if ti > length(tt)
            break
        end
    end
    
    while ti <= length(tt)
        out[ti,:] .= traj.states[end]
        ti += 1
    end
    
    out
end

Base.convert(::Type{Trajectory}, sol::SciMLBase.ODESolution) = convert(Trajectory{SVector{length(sol.prob.u0), Species}}, sol)

function Base.convert(::Type{Trajectory{State}}, sol::SciMLBase.ODESolution) where {State}
    return Trajectory(sol.t, convert(Vector{State}, sol.u), length(sol.prob.u0))
end

mutable struct Trajectory{State <: AbstractVector{Species}}
    times::Vector{Float64}
    states::Vector{State}
    n_species::Int64
end

function clean_traj(traj::Trajectory{State}) where {State}
    new_times = [ traj.times[1] ]
    new_states = State[ traj.states[1] ]
    
    
    for i in 2:length(traj.times)
        if traj.states[i] == traj.states[i-1]
            continue
        end
        
        push!(new_times, traj.times[i])
        push!(new_states, traj.states[i])
    end
    
    push!(new_times, traj.times[end])
    push!(new_states, traj.states[end])
    
    ret = Trajectory{State}(new_times, new_states, traj.n_species)
end

StatsBase.mean(traj::Trajectory) = sum(traj.states[1:end-1] .* diff(traj.times)) / traj.times[end]
meansq(traj::Trajectory) = sum(map.(x -> x^2, traj.states[1:end-1]) .* diff(traj.times)) / traj.times[end]
StatsBase.var(traj::Trajectory) = meansq(traj) .- mean(traj) .^ 2
StatsBase.std(traj::Trajectory) = sqrt.(var(traj))

###################### Reaction systems ######################

mutable struct ReactionSystemInfo{T, RFT}
    rs::ReactionSystem
    ps::Vector{T}
    jsys::JumpSystem
    ratefuncs::RFT
    haveivdeps::Vector{Bool}
    netstoich::Matrix{Int}
    speciesmap::Dict{Any,Int}
    paramsmap::Dict{Any,Int}
end

function ReactionSystemInfo(rs::ReactionSystem; ps::AbstractVector{T}) where {T}
    @assert isempty(MT.get_systems(rs)) && isempty(MT.get_observed(rs))
    
    jsys = convert(JumpSystem, rs)
    
    ratefuncs = tuple((eval(MT.generate_rate_function(jsys, Catalyst.jumpratelaw(eq))) for eq in MT.get_eqs(rs))...)
    
    return ReactionSystemInfo(rs, convert(Vector{T}, ps), jsys, ratefuncs, 
                              getivdeps(rs),
                              prodstoichmat(rs) .- substoichmat(rs),
                              convert(Dict{Any,Int}, speciesmap(rs)), 
                              convert(Dict{Any,Int}, paramsmap(rs)))
end

function getivdeps(rs::ReactionSystem)
    rxvars = []
    haveivdeps = Bool[]
    for rx in MT.get_eqs(rs)
        empty!(rxvars)
        (rx.rate isa MT.Symbolic) && MT.get_variables!(rxvars, rx.rate)
        push!(haveivdeps, any(rxvar -> isequal(rxvar, MT.get_iv(rs)), rxvars))
    end
    
    haveivdeps
end

function ReactionSystemInfo(rsi::ReactionSystemInfo; ps::AbstractVector{T}) where {T}
    return ReactionSystemInfo(rsi.rs, convert(Vector{T}, ps), rsi.jsys, rsi.ratefuncs, rsi.haveivdeps,
                              rsi.netstoich, rsi.speciesmap, rsi.paramsmap)
end

function simulate_trajs(sys::ReactionSystem, n; u0=zeros(Int, numspecies(sys)), tmax, ps)
    prob = DiscreteProblem(sys, u0, (0.0, tmax), ps)
    jump_prob = JumpProblem(sys, prob, Direct())

    ret = Trajectory[]
    for i in 1:n
        sol = solve(jump_prob, SSAStepper());
        push!(ret, convert(Trajectory, sol))
    end
    
    ret
end


function simulate_trajs(rsi::ReactionSystemInfo, n; u0=zeros(Int, numspecies(rsp.rs)), tmax, ps=rsi.ps)
    prob = DiscreteProblem(rsi.jsys, u0, (0.0, tmax), ps)
    jump_prob = JumpProblem(rsi.jsys, prob, Direct())

    ret = Trajectory[]
    for i in 1:n
        sol = solve(jump_prob, SSAStepper());
        push!(ret, convert(Trajectory, sol))
    end
    
    ret
end


function simulate_traj(sys; kwargs...)
    simulate_trajs(sys, 1; kwargs...)[1]
end

##########


print_marg(marg::Species) = string(marg)    
print_marg(marg::AbstractVector{Species}) = string(tuple(sort(marg)...))
    
get_maximum(traj::Trajectory, marg::Marginal) = maximum(extract_marg(state, marg) for state in traj.states) 

function get_traj_marginal(trajs::AbstractVector{<: Trajectory}, 
                           tt::Vector{Float64}, marg::Marginal, nmax::Int64)
    return get_traj_marginals(trajs, tt, (marg,), [nmax])[1]
end

function get_traj_marginals(trajs::AbstractVector{<: Trajectory}, 
                            tt::Vector{Float64}, margs::NTuple{N,Marginal}, 
                            nmax::Vector{Int64})::NTuple{N,Array{Int64}} where {N}
    n_margs = length(margs)
    ret = tuple([ zeros(Int64, length(tt), nmax[i]+1) for i in range(1, stop=n_margs) ]...)
    
    for (i, traj) in enumerate(trajs)
        for (j, t) in enumerate(tt)
            idx = findlast(x -> x <= t, traj.times)
            if idx == nothing
                continue
            end
            
            for (k, marg) in enumerate(margs)
                count = extract_marg(traj.states[idx], marg)
                
                if count <= nmax[k]
                    ret[k][j,count+1] += 1
                end
            end
        end
    end
    
    return ret
end