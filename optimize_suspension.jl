# Optimization Function
# Jeff Robinson - jbrobin@stanford.edu
include("suspension_model_objective.jl")
include("weighted_sum.jl")
include("nelder_mead.jl")
include("covariance_matrix_adaptation.jl")
include("cyclic_coordinate_descent.jl")
include("generalized_pattern_search.jl")


function constraints(state_vec)
  c = []

  # Damping Ratio
  ζ_crit = 2*sqrt(state_vec[1]*default_inputs.m1)
  for i = 1:4
    ζ = state_vec[i+2]/ζ_crit
    if ζ > 1.0
      push!(c, (ζ - 1.0)*100)
    elseif ζ < 0.2
      push!(c, (0.2 - ζ)*100)
    else
      push!(c, 0.0)
    end
  end
  
  return c
end

## Quadratic Penalty, Inequality <= 0 constraints only ("Algorithms For Optimization" Equation 10.41, Kochenderfer & Wheeler) ##
function P_quad(constraint_values)
  P_quadratic = 0.0
  for i = 1:length(constraint_values)
      P_quadratic += max(constraint_values[i], 0.0)^2
  end
  return P_quadratic
end

## Count Penalty, Inequality <= 0 constraints only ("Algorithms For Optimization" Equation 10.39, Kochenderfer & Wheeler) ##
function P_count(constraint_values)
  P_count = 0.0
  for i = 1:length(constraint_values)
      if constraint_values[i] > 0.0
          P_count += 1.0
      end
  end
  return P_count
end

function penalties(x)
  constraint_vals = constraints(x)
  return P_quad(constraint_vals) + P_count(constraint_vals)
end


function optimize_suspension(
  method = "none", 
  max_n_evals = 100, 
  weights = [0.99,0.01]
)

  function f(x, n_evals = 0, max_n_evals = max_n_evals, fweights = weights)
    f = weighted_sum(suspension_model_objective, x, fweights)
    p = penalties(x)
    n_evals += 1
    println(n_evals, " / ", max_n_evals)
    return f + p, n_evals
  end

  # state_vec = [k1_0, [kN/m], 
               # k2_0, [kN/m], 
               # bRH, [kN-s/m], 
               # bRL, [kN-s/m], 
               # bCL, [kN-s/m], 
               # bCH, [kN-s/m], 
               # y_dotcritR, [m/s], 
               # y_dotcritC, [m/s]]
  defaults = [6.0, 20.0, 1.0, 1.0, 1.0, 1.0, 2.0, -2.0]
  n_dims = length(defaults)

  if method == "CCD"
    x_best, x_log = cyclic_coordinate_descent(
      f, 
      defaults, 
      max_n_evals, 
      evals_per_search = 20
    )
  elseif method == "NMS" # Nelder-Mead Simplex
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

  elseif method == "GPS"
    D = [basis(i, n_dims) for i = 1:n_dims]
    D = vcat(D, -D)
    x_best = generalized_pattern_search(
      f, 
      defaults, 
      [2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0], # α - step size vector
      D, 
      max_n_evals
      # ϵ = 0.01, γ=0.5
      )

  elseif method == "CMA" # Covariance Matrix Adaptation
    x_best = covariance_matrix_adaptation(
      f, 
      defaults, 
      max_n_evals
      # σ = 1.0,
      # m = 4 + floor(Int, 3*log(length(x))),
      # m_elite = div(m, 2),
    )

  elseif method == "none"
    x_best = defaults

  end

  y_best = suspension_model_objective(x_best, plotting = true)
  println("Ground Following (mean distance from tire to ground): ", y_best[1], "\nRide Comfort (RMS vertical acceleration of bike frame): ", y_best[2])
  return x_best
end
