## Particle Swarm Optimization (Kochenderfer & Wheeler Algorithm 9.11-12) ##
mutable struct particle
  x
  v
  x_best
end
function particle_swarm_optimization(f, population, max_n_evals;
  w = 1,
  c1 = 1,
  c2 = 1
  )
  n_dims, m = length(population[1].x), length(population)
  k_max = floor(Int32, (max_n_evals - m)/(2*m))
  x_best, y_best = copy(population[1].x_best), Inf
  for P in population
    y = f(P.x)
    if y < y_best
      x_best[:], y_best = P.x, y
    end
  end
  for k in 1:k_max
    for P in population
      r1, r2 = rand(n_dims), rand(n_dims)
      P.x += P.v
      P.v = w*P.v + c1*r1.*(P.x_best - P.x) + c2*r2.*(x_best - P.x)
      y = f(P.x)
      if y < y_best
        x_best[:], y_best = P.x, y
      end
      if y < f(P.x_best)
        P.x_best[:] = P.x
      end
    end
  end
  return x_best
end