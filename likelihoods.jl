include("./cme.jl")
using Distributions

struct Transition{T} 
    curr::T
    next::T
    tprev::Float64
    tjump::Float64
end

function get_transition_rate(rsi::ReactionSystemInfo{T}, ps, rf, 
                             stoich::AbstractVector{Int}, haveivdep::Bool,
                             trans::Transition) where {T}
    if haveivdep
        rate_next = 1. * rf(trans.curr, ps, trans.tjump)
        lambda_int = 0.5 * (rate_next + rf(trans.curr, ps, trans.tprev)) * (trans.tjump - trans.tprev)
    else
        rate_next = 1. * rf(trans.curr, ps, trans.tjump)
        lambda_int = rate_next * (trans.tjump - trans.tprev)      
    end
    
    @inbounds for i in 1:length(stoich)
        if stoich[i] != trans.next[i] - trans.curr[i]
            rate_next = zero(rate_next)        
            break
        end
    end
    
    return (lambda_int, rate_next)
end

@generated function get_transition_rate(rsi::ReactionSystemInfo{T,RFT}, ps, trans::Transition) where {T,RFT}
    ex = :(lambda_int = zero(T); rate_next = zero(T))
    
    for (i, t) in enumerate(RFT.parameters)
        ex = quote
            $ex
            
            lambda_int_i, rate_next_i = get_transition_rate(rsi, ps, rsi.ratefuncs[$i], 
                                                            (@view rsi.netstoich[:,$i]), 
                                                            rsi.haveivdeps[$i], trans)
            
            lambda_int += lambda_int_i
            rate_next += rate_next_i
        end
    end
    
    ex = :($ex; return (lambda_int, rate_next))
end


function Distributions.logpdf(rsi::ReactionSystemInfo{T}, traj::Trajectory; ps=rsi.ps) where {T}
    ret = zero(T)
    dts = diff(traj.times)
    n_species = length(first(traj.states))
    
    curr_t = 0.0
    
    for i in 1:length(dts)-1
        dt = dts[i]
        curr = traj.states[i]
        next = traj.states[i+1]
        
        if next == curr
            curr_t += dt
            continue
        end
        
        trans = Transition(curr, next, traj.times[i+1] - curr_t - dt, traj.times[i+1])
        
        lambda_int, rate_next = get_transition_rate(rsi, ps, trans)
        
        ret += log(rate_next) - lambda_int 
        
        rate_next == 0 && @warn "Impossible transition: $trans" 
        isfinite(ret) || return ret
        
        curr_t = 0.
    end
    
    if traj.states[end-1] == traj.states[end]
        trans = Transition(traj.states[end], traj.states[end], traj.times[end-1] - curr_t, traj.times[end])
        lambda_int, rate_next = get_transition_rate(rsi, ps, trans)
        
        dt = trans.tjump - trans.tprev
        
        ret -= lambda_int 
    else
        @warn "trajectory seems truncated"
    end
    
    return ret
end



@generated function Distributions.logpdf(rsi::ReactionSystemInfo{T}, traj::Trajectory, reaction::Val{N}; ps=rsi.ps) where {T,N}
    quote
        ret = zero(T)
        
        rf = rsi.ratefuncs[$N]
        Si = (@view rsi.netstoich[:,$N])
        haveivdep = rsi.haveivdeps[$N]

        for i in 1:length(traj.times)-2
            trans = Transition(traj.states[i], traj.states[i+1], traj.times[i], traj.times[i+1])

            lambda_int, rate_next = get_transition_rate(rsi, ps, rf, Si, haveivdep, trans)

            ret -= lambda_int 
            if !iszero(rate_next)
                ret += log(rate_next)
            end
        end

        if traj.states[end-1] == traj.states[end]
            trans = Transition(traj.states[end], traj.states[end], traj.times[end-1], traj.times[end])
            lambda_int, rate_next = get_transition_rate(rsi, ps, rf, Si, haveivdep, trans)

            ret -= lambda_int 
        else
            @warn "trajectory seems truncated"
        end

        ret
    end
end

Distributions.logpdf(rsi::ReactionSystemInfo, traj::Trajectory, reaction::Int; kwargs...) = logpdf(rsi, traj, Val(reaction); kwargs...)

function Distributions.logpdf(rsi::ReactionSystemInfo, traj::Trajectory, reactions::AbstractVector{Int}; kwargs...)
    sum(logpdf(rsi, traj, Val(reaction); kwargs...) for reaction in reactions)
end

function Distributions.logpdf(rsi::ReactionSystemInfo, traj::Trajectory, reactions::Tuple; kwargs...)
    sum(logpdf(rsi, traj, reaction; kwargs...) for reaction in reactions)
end