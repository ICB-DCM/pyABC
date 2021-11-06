module SIR

# Define reaction network
using Catalyst
sir_model = @reaction_network begin
    r1, S + I --> 2I
    r2, I --> R
end r1 r2

# ground truth parameter
p = (0.0001, 0.01)
# initial state
u0 = [999, 1, 0]
# time span
tspan = (0.0, 250.0)
# formulate as discrete problem
prob  = DiscreteProblem(sir_model, u0, tspan, p)

# formulate as Markov jump process
using DiffEqJump
jump_prob = JumpProblem(
    sir_model, prob, Direct(), save_positions=(false, false),
)

"""
Simulate model for parameters `10.0.^par`.
"""
function model(par)
    p = 10.0.^((par["p1"], par["p2"]))
    sol = solve(remake(jump_prob, p=p), SSAStepper(), saveat=2.5)
    return Dict("t"=>sol.t, "u"=>sol.u)
end

# observed data
print(p)
observation = model(Dict("p1"=>log10(p[1]), "p2"=>log10(p[2])))

"""
Distance between model simulations or observed data `y` and `y0`.
"""
function distance(y, y0)
    u, u0 = y["u"], y0["u"]
    if length(u) != length(u0)
        throw(AssertionError("Dimension mismatch"))
    end
    return sum((u .- u0).^2) / length(u0)
end

end  # module
