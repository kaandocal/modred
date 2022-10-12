using Catalyst
using StaticArrays

include("../reduce.jl")

# Original system
sys = @reaction_network begin
    k0, G_u --> G_u + M
    ks, M --> M + P
    (k3 / Ω, km3), P + E <--> EP
    k4, EP --> E
    kdm, M --> 0
    (km2, k2 / Ω), G_b <--> GP + P
    (km1, k1 / Ω), GP <--> G_u + P
    k0, GP --> GP + M
end Ω k0 ks k3 km3 k4 kdm k1 km1 k2 km2

gt_params = [ 1000., 50, 0.0045, 0.1, 10, 10, 0.01, 0.001, 100, 1000, 1 ]
u0 = [ 1, 0, 0, 10, 0, 0, 0 ]

# Model I
sys_noE = @reaction_network begin
    k0, G_u --> G_u + M
    ks, M --> M + P
    kdp, P --> 0
    kdm, M --> 0
    (km2, k2 / Ω), G_b <--> GP + P
    (km1, k1 / Ω), GP <--> G_u + P
    k0, GP --> GP + M
end Ω k0 ks kdp kdm k1 km1 k2 km2

sys_cpt_noE = reducereacsys(sys, [4])[1]

u0_noE = [ 1, 0, 0, 0, 0 ]
u0_cpt_noE = [ 1, 0, 0, 0, 0, 0 ]

# Model II
sys_1Gp = @reaction_network begin
    k0, G_u --> G_u + M
    ks, M --> M + P
    (k3 / Ω, km3), P + E <--> EP
    k4, EP --> E
    kdm, M --> 0
    (vg / (Kg + P), k1 / Ω), G_b <--> G_u + P
    keff, G_b --> G_b + M
end Ω k0 ks k3 km3 k4 kdm k1 vg Kg keff

u0_1Gp = [ 1, 0, 0, 10, 0, 0 ]

# Model I + II
sys_noE1Gp = @reaction_network begin
    k0, G_u --> G_u + M
    ks, M --> M + P
    kdp, P --> 0
    kdm, M --> 0
    (vg / (Kg + P), k1 / Ω), G_b <--> G_u + P
    keff, G_b --> G_b + M
end Ω k0 ks kdp kdm k1 vg Kg keff

sys_cpt_noE1Gp = reducereacsys(sys, [4])[1]

u0_noE1Gp = [ 1, 0, 0, 0 ]
u0_cpt_noE1Gp = [ 1, 0, 0, 0, 0, 0 ]

# Model IV
sys_1G = @reaction_network begin
    k0, G_u --> G_u + M
    ks, M --> M + P
    (k3 / Ω, km3), P + E <--> EP
    k4, EP --> E
    kdm, M --> 0
    (km2, vg / (Kg + P)), G_b <--> G_u + 2P
end Ω k0 ks k3 km3 k4 kdm k1 km1 vg Kg km2

u0_1G = [ 1, 0, 0, 10, 0, 0 ]

# Model V
sys_noG = @reaction_network begin
    keff * (1 + α * P) / (1 + α * P + β * P^2), 0 --> M
    ks, M --> M + P
    (k3 / Ω, km3), P + E <--> EP
    k4, EP --> E
    kdm, M --> 0
end Ω keff α β ks k3 km3 k4 kdm

sys_cpt_noG = @reaction_network begin
    k0 * (G != 2), 0 --> M
    ks, M --> M + P
    (k3 / Ω, km3), P + E <--> EP
    k4, EP --> E
    kdm, M --> 0
    (km2 * (G == 2) / 2, k2 / Ω * (G == 1)), G <--> P
    (km1 * (G == 1), k1 / Ω * (G == 0)), G <--> P
end Ω k0 ks k3 km3 k4 kdm k1 km1 k2 km2

u0_noG = [ 0, 0, 10, 0 ]

# Utility functions for marginalising (projecting) trajectories
marg_noE(s) = @SVector [ s[1], s[2], s[3] + s[5], s[6], s[7] ]
marg_traj_noE(traj::Trajectory) = Trajectory(traj.times, map(marg_noE, traj.states), 5)

marg_1Gp(s) = @SVector [ s[1], s[2], s[3] + s[6], s[4], s[5], s[6] + s[7] ]
marg_traj_1Gp(traj::Trajectory) = Trajectory(traj.times, map(marg_1Gp, traj.states), 6)

marg_noE1Gp(s) = @SVector [ s[1], s[2], s[3] + s[5] + s[6], s[6] + s[7] ]
marg_traj_noE1Gp(traj::Trajectory) = Trajectory(traj.times, map(marg_noE1Gp, traj.states), 4)

marg_1G(s) = @SVector [ s[1] + s[7], s[2], s[3] + s[7], s[4], s[5], s[6] ]
marg_traj_1G(traj::Trajectory) = Trajectory(traj.times, map(marg_1G, traj.states), 6)

marg_noG(s) = @SVector [ s[2], s[3] + s[7] + 2 * s[6], s[4], s[5] ]
marg_traj_noG(traj::Trajectory) = Trajectory(traj.times, map(marg_noG, traj.states), 4)
