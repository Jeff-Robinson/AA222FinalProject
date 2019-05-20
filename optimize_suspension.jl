# Optimization Function
# Jeff Robinson - jbrobin@stanford.edu
include("nelder_mead.jl")
include("suspension_model_objective.jl")
include("weighted_sum.jl")

function optimize_suspension(
  method = "none", 
  max_n_evals = 10, 
  weights = [0.9,0.1]
)
function f(x, fweights = weights)
  f = weighted_sum(suspension_model_objective, x, fweights)
  return f
end
defaults = [6000.0, 20000.0, 1000.0, 1000.0, 1000.0, 1000.0, 2.0, -2.0]
n_dims = length(defaults)

if method == "Nelder-Mead"
  S = []
  for i = 1:n_dims + 1
      S_i = [defaults[1]*rand()*2,
             defaults[2]*rand()*2,
             defaults[3]*rand()*2,
             defaults[4]*rand()*2,
             defaults[5]*rand()*2,
             defaults[6]*rand()*2,
             rand()*2,
             -rand()*5
      ]
      push!(S, S_i)
  end
  x_best = nelder_mead(
    f,
    S,
    max_n_evals
  ) #; α=1.0, β=2.0, γ=0.5)
  
elseif method == "none"
  x_best = defaults
end

return x_best

end