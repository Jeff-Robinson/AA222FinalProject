# Optimization Function
# Jeff Robinson - jbrobin@stanford.edu
include("nelder_mead.jl")
include("suspension_model_objective.jl")
include("weighted_sum.jl")

@time function optimize_suspension(
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
  S = [defaults]
  for i = 1:n_dims
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

y_best = suspension_model_objective(x_best, plotting = true)
println("Ground Following (mean distance from tire to ground): ",y_best[1], "\nRide Comfort (RMS vertical acceleration of bike frame): ", y_best[2])
return x_best
end

# Radial Basis Functions (Kochenderfer & Wheeler Algorithm 14.5)
# radial_linear(r) = r
# radial_quadratic(r) = r^2
# radial_cubic(r) = r^3
# radial_gaussian(r, σ=1000.0) = exp(-r^2/(2*σ^2))
# radial_invgaussian(r, σ=1000.0) = 1-exp(-r^2/(2*σ^2))
# radial_bases(C, x; ψ=radial_invgaussian, p=2) = sum([ψ(norm(x - c[1], p)) for c in C])