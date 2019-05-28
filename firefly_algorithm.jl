## Firefly Algorithm (Kochenderfer & Wheeler Algorithm 9.13) ##
function firefly(f, population, max_n_evals;
  β = 1,
  α = 0.1,
  brightness = r -> exp(-r^2)
  )
  n, m = length(population[1]), length(population)
  # k_max = floor(Int32, (max_n_evals - m)/((2*m)^2))
  N = MvNormal(Matrix(1.0I, n, n))
  # for k in 1:k_max
  x_best, y_best = population[1], Inf
  x_log = [x_best]
  evals_log = [NUM_FXN_EVALS]
  while true
    for a in population, b in population
      ya, yb = f(a), f(b)
      if yb < ya
        r = norm(b-a)
        a[:] += β*brightness(r)*(b-a) + α*rand(N)
      end
      if yb < y_best
        x_best, y_best = b, yb
        push!(x_log, x_best)
        push!(evals_log, NUM_FXN_EVALS)
      elseif ya < y_best
        x_best, y_best = a, ya
        push!(x_log, x_best)
        push!(evals_log, NUM_FXN_EVALS)
      end

      if NUM_FXN_EVALS >= max_n_evals
        break
      end
    end

    if NUM_FXN_EVALS >= max_n_evals
      break
    end
  end
  # ys = [f(x) for x in population]
  # return population[argmin(ys)]
  return x_best, x_log, evals_log
end