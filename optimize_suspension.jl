# Optimization Function
# Jeff Robinson - jbrobin@stanford.edu

include("suspension_model_objective.jl")
include("nelder_mead.jl")
include("covariance_matrix_adaptation.jl")
include("cyclic_coordinate_descent.jl")
include("generalized_pattern_search.jl")
include("adaptive_simulated_annealing.jl")
include("particle_swarm_optimization.jl")
include("firefly_algorithm.jl")

function constraints(state_vec)
  c = zeros(8)

  # Signs
  for i = 1:7
    if state_vec[i] <= 0
      c[i] -= state_vec[i]*default_inputs.state_orders[i]
    end
  end
  if state_vec[8] >= 0
    c[8] += state_vec[8]*default_inputs.state_orders[8]
  end

  # Damping Ratio
  ζ_crit = 2*sqrt(abs(state_vec[1]*default_inputs.state_orders[1])*default_inputs.m1)
  for i = 3:6
    ζ = state_vec[i]*default_inputs.state_orders[i]/ζ_crit
    if ζ > 1.0
      c[i] += (ζ - 1.0)
    elseif ζ < 0.2
      c[i] += (0.2 - ζ)
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
  p_val = P_quad(constraint_vals) + P_count(constraint_vals)
  return 100*p_val
end


function weighted_sum(f, x, weights)
combined_f = sum(f(x).*weights)
return combined_f
end


function optimize_suspension(;
  method = "none", 
  max_n_evals = 100, 
  weights = [0.99,0.01]
)

  global NUM_FXN_EVALS = 0
  function f(x, max_n_evals = max_n_evals, fweights = weights)
    f = weighted_sum(suspension_model_objective, x, fweights)
    p = penalties(x)
    NUM_FXN_EVALS += 1
    println(NUM_FXN_EVALS, " / ", max_n_evals)
    return f + p
  end

  # state_vec = [k1_0, [kN/m], 
               # k2_0, [kN/m], 
               # bRH, [kN-s/m], 
               # bRL, [kN-s/m], 
               # bCL, [kN-s/m], 
               # bCH, [kN-s/m], 
               # y_dotcritR, [m/s], 
               # y_dotcritC, [m/s]]
  defaults = default_inputs.state_vec
  n_dims = length(defaults)
  step_sizes = [2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]

  if method == "CCD"
    x_best, x_log, evals_log = cyclic_coordinate_descent(
      f, 
      defaults, 
      max_n_evals, 
      evals_per_search = 20
    )
    
  elseif method == "NMS" # Nelder-Mead Simplex
    S = [defaults]
    for i = 1:n_dims
        # S_i = [defaults[1]*rand()*2,
        #       defaults[2]*rand()*2,
        #       defaults[3]*rand()*2,
        #       defaults[4]*rand()*2,
        #       defaults[5]*rand()*2,
        #       defaults[6]*rand()*2,
        #       rand()*2,
        #       -rand()*5
        # ]
        S_i = defaults.+step_sizes.*rand(Uniform(-1.0,1.0), n_dims)
        push!(S, S_i)
    end
    x_best, x_log, evals_log = nelder_mead(
      f,
      S,
      max_n_evals
    ) #; α=1.0, β=2.0, γ=0.5)

  elseif method == "GPS"
    D = [basis(i, n_dims) for i = 1:n_dims]
    D = vcat(D, -D)
    x_best, x_log, evals_log = generalized_pattern_search(
      f, 
      defaults, 
      step_sizes, # α - step size vector
      D, 
      max_n_evals
      # ϵ = 0.01, γ=0.5
    )

  elseif method == "CMA" # Covariance Matrix Adaptation
    x_best, x_log, evals_log = covariance_matrix_adaptation(
      f, 
      defaults, 
      max_n_evals
      # σ = 1.0,
      # m = 4 + floor(Int, 3*log(length(x))),
      # m_elite = div(m, 2),
    )

  elseif method == "ASA"
    x_best, x_log, evals_log = adaptive_simulated_annealing(
      f, 
      defaults, 
      step_sizes, # v, step size vector
      100,# t, initial temperature
      0.01, # ϵ
      max_n_evals
      # ns = 20,
      # nϵ = 4,
      # nt = max(100,5*length(x)),
      # γ = 0.85,
      # c = fill(2, length(x))
    )

  elseif method == "PSO"
    population = [particle(defaults, zeros(n_dims), defaults)]
    pop_size = 20
    for i = 2:pop_size
      particle_x = defaults.+step_sizes.*rand(Uniform(-1.0,1.0), n_dims)
      push!(population, particle(particle_x, zeros(n_dims), defaults))
    end
    x_best, x_log, evals_log = particle_swarm_optimization(
      f, 
      population, 
      max_n_evals
      # w = 1,
      # c1 = 1,
      # c2 = 1
    )

  elseif method == "firefly"
    flies = [defaults]
    pop_size = 5
    for i = 2:pop_size
      push!(flies, defaults.+step_sizes.*rand(Uniform(-1.0,1.0), n_dims))
    end
    x_best, x_log, evals_log = firefly(
      f, 
      flies, 
      max_n_evals
      # β = 1,
      # α = 0.1,
      # brightness = r -> exp(-r^2)
    )

  elseif method == "none"
    x_best = defaults

  end

  y_best = suspension_model_objective(x_best)
  println("Ground Following (mean distance from tire to ground): ", y_best[1], "\nRide Comfort (RMS vertical acceleration of bike frame): ", y_best[2])

  
  return x_best
end
