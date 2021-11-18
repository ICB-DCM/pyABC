module Normal

using Random

n_y = 4
std = 0.1

function model(par)
    # just some additive Gaussian noise
    return Dict("y" => par["p"] .+ std .* randn(n_y))
end

# observed data
gt_par = Dict("p" => 0)
observation = model(gt_par)

function distance(y, y0)
    # just a Euclidean distance
    return sum((y["y"] .- y0["y"]) .^2)
end

end  # module
