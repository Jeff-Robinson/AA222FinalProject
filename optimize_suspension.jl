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

using JLD

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



method_list = ["CCD", "NMS", "GPS", "CMA", "ASA", "PSO", "firefly"]
function optimize_suspension(;
  method, 
  max_n_evals, 
  weights = [0.99,0.01],
  save = false
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
    pop_size = 15 # n_iters = (max_n_evals - pop_size)/(2*pop_size)
    # pop_size = max_n_evals/(2*n_iters + 1)
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
    pop_size = 5 # n_iters = (max_n_evals - pop_size)/(2*pop_size)^2
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

  end

  y_best = suspension_model_objective(x_best)
  println("Ground Following (mean distance from tire to ground): ", y_best[1], "\nRide Comfort (RMS vertical acceleration of bike frame): ", y_best[2])

  if save == true
    save_name = new_savefile_name(method)
    savefile = jldopen(save_name, "w")
    write(savefile, "method", method)
    write(savefile, "max_n_evals", max_n_evals)
    write(savefile, "weights", weights)
    write(savefile, "x_best", x_best)
    write(savefile, "y_best", y_best)
    write(savefile, "x_log", x_log)
    write(savefile, "evals_log", evals_log)

    # read(savefile)
    # get(read(savefile), "a", "error")
  end
  return x_best
end



function new_savefile_name(method, number = 1)
  savefile_name = "$(method)_savefile_$(number).jld"
  try
    jldopen(savefile_name, "r")
    savefile_name(method, number += 1)
  catch
    return savefile_name
  end
end



function existing_savefile_names(method, number = 1, savefile_names = [])
  savefile_name = "$(method)_savefile_$(number).jld"
  try
    jldopen(savefile_name, "r")
    push!(savefile_names, savefile_name)
    return existing_savefile_names(method, number += 1, savefile_names)
  catch
    return savefile_names   
  end
end



function all_existing_savefile_names()
  existing_saves = []
  for name in method_list
    append!(existing_saves, existing_savefile_names(name))
  end
  return existing_saves
end



function optimize_all_algorithms()
  for method in method_list
    println(method)
    @time optimize_suspension(method = method, 
    max_n_evals = 320, 
    weights = [0.5,0.5],
    save = true
    )
  end
end



function generate_y_logs()
  savefile_names = all_existing_savefile_names()
  for name in savefile_names
    savefile = jldopen(name, "w")
    x_log = get(read(savefile), "x_log", "error")
    weights = get(read(savefile), "weights", "error")
    println(name, ": x_log length = ", length(x_log))
    y_log = []
    y_log_weighted = []
    for i in 1:length(x_log)
      println(i," / ", length(x_log))
      push!(y_log, suspension_model_objective(x_log[i]))
      push!(y_log_weighted, sum(y_log[i].*weights))
    end
    write(savefile, "y_log", y_log)
    write(savefile, "y_log_weighted", y_log_weighted)
    close(savefile)
  end
end



function convergence_plot()
  savefile_names = all_existing_savefile_names()
  fig = figure()
  ax = fig.subplots()
  ax.set_title("Convergence Plot, All Algorithms")
  ax.set_xlabel("Number of Function Evaluations")
  ax.set_ylabel("Weighted Objective Value")
  legend = ()
  line_styles = ["-","--"]
  for i in 1:length(savefile_names)
    savefile = jldopen(savefile_names[i], "r")
    method = get(read(savefile), "method", "error")
    weights = get(read(savefile), "weights", "error")
    evals_log = get(read(savefile), "evals_log", "error")
    y_log_weighted = get(read(savefile), "y_log_weighted", "error")

    line_color = 0.5*(i/length(savefile_names) + 1)
    ax.plot(evals_log, y_log_weighted, 
            color = (line_color, line_color, line_color), 
            linestyle = line_styles[mod(i+1, 2)+1])
    legend = (legend..., "$(method), weights = $(weights)")
  end
  ax.legend(legend)
end