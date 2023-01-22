# Demo: Use Alg. 1 to reduce a telegraph model
using Catalyst
using JumpProcesses
using Random
using ForwardDiff

# Telegraph model
rn = @reaction_network begin
    σ_on * (1 - G), 0 --> G
    σ_off, G --> 0
    ρ, G --> G + P
end σ_on σ_off ρ

u0 = [ 0., 0. ]        # Start with mRNA and gene in state 0 (= off)
tmax = 100.
ps = [ 1., 1., 2.5 ]   # System parameters

dprob = DiscreteProblem(rn, u0, (0., tmax), ps)
jprob = JumpProblem(rn, dprob, Direct())

# Reduced model
rn_red = @reaction_network begin
    ρ, 0 --> P
end ρ

# Computed the log likelihood of a trajectory under the reduced model
# using Eq. 4.
# For this simple model we could have used Eq. 6 directly, but this
# is more instructive
function loglikelihood_reduced(traj, ps)
    ρ = ps[]

    ret = zero(ρ)
    for i in 1:length(traj.t)-1
        dt = traj.t[i+1] - traj.t[i]
        ret -= dt * ρ

        # Check if mRNA is produced and record this in the result
        if traj.u[i+1] > traj.u[i]
            ret += log(ρ)
        end
    end

    ret
end

### Model reduction ###

# Step 1: Generate sample data from the full model
tmax = 2e4
N = 20
trajs = []
for i in 1:N
    sol = solve(jprob, SSAStepper())
    traj = (t = sol.t, u = map(x -> x[2], sol.u))
    push!(trajs, traj)
end

# Step 2: Optimise!
# We want to minimise the average negative log-likelihood
loss(ps) = -sum(traj -> loglikelihood_reduced(traj, ps), trajs) / N

curr_ps = rand(1)
η = 0.001            # Learning rate
nsteps = 100        # Number of iteration steps

for i in 1:nsteps
    grad = ForwardDiff.gradient(loss, curr_ps)
    curr_ps .-= η * grad
end

println("Estimated parameters: ρ = $(curr_ps[])")
println("Analytical prediction: ρ = $(ps[1] / (ps[1] + ps[2]) * ps[3])")
