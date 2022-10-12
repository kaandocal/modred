function add_bursty_reactions!(sys, ρ_u, ρ_b, b; Nmax=15)
    Pterm = sys.states[2]
    
    for j in 2:Nmax
        rxu = Reaction(ρ_u * (b^j) / (1 + b)^(j+1), 
                       sys.eqs[1].substrates, sys.eqs[1].products, 
                       [1], [1,j], [ Pterm => j ], 
                       false)

        rxb = Reaction(ρ_b * (b^j) / (1 + b)^(j+1), 
                       sys.eqs[2].substrates, sys.eqs[2].products, 
                       [1], [1,j], [ Pterm => j ], 
                       false)

        addreaction!(sys, rxu)
        addreaction!(sys, rxb)
    end
    
    sys
end

function add_bursty_reactions_qea!(sys, ρ_u, ρ_b, b; Nmax=15)
    Pterm = sys.states[1]
    
    @parameters t
    @variables σ_b P(t)
    for j in 2:Nmax
        rx = Reaction(Symbolics.value((ρ_u * 400 + ρ_b * σ_b * P) / (400 + σ_b * P) * (b^j) / (1 + b)^(j+1)), 
                       sys.eqs[1].substrates, sys.eqs[1].products, 
                       Int[], [j], [ Pterm => j ], 
                       false)

        addreaction!(sys, rx)
    end
    
    sys
end

@parameters σ_b

sys = @reaction_network begin
    0.3 * (2/9), G_u --> G_u + P
    105. * (2/9), G_b --> G_b + P
    1.0, P --> 0
    (σ_b * P, 400), G_u <--> G_b
end σ_b

add_bursty_reactions!(sys, 0.3, 105., 2);

sys_red = @reaction_network begin
    0.3 * (2/9), G_u --> G_u + P
    105. * (2/9), G_b --> G_b + P
    1.0, P --> 0
    (σ_b, 400), G_u <--> G_b
end σ_b

add_bursty_reactions!(sys_red, 0.3, 105., 2);

rsi = ReactionSystemInfo(sys, ps=[2.5])
rsi_red = ReactionSystemInfo(sys_red, ps=[2.5]);