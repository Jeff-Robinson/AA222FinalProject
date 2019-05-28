using Distributions

## Adaptive Simulated Annealing (Kochenderfer & Wheeler Algorithm 8.6) ##
function adaptive_simulated_annealing(f, x, v, t, ϵ, max_n_evals;
  ns = 20,
  nϵ = 4,
  nt = max(100,5*length(x)),
  γ = 0.85,
  c = fill(2, length(x))
  )

  y = f(x)
  x_best, y_best = x, y
  y_arr, n_dims, U = [], length(x), Uniform(-1.0,1.0)
  a, counts_cycles, counts_resets = zeros(n_dims), 0, 0

  while true
    for i in 1:n_dims
      xp = x + basis(i, n_dims)*rand(U)*v[i]
      yp = f(xp)
      Δy = yp - y
      if Δy < 0 || rand() < exp(-Δy/t)
        x, y = xp, yp
        a[i] += 1
        if yp < y_best
          x_best, y_best = xp, yp
        end
      end

      if NUM_FXN_EVALS >= max_n_evals
        return x_best
      end
    end

    counts_cycles += 1
    counts_cycles >= ns || continue

    counts_cycles = 0
    corana_update!(v, a, c, ns)
    fill!(a, 0)
    counts_resets += 1
    counts_resets >= nt || continue
    t *= γ
    counts_resets = 0
    push!(y_arr, y)

    if !(length(y_arr) > nϵ && 
      y_arr[end] - y_best <= ϵ &&
      all(abs(y_arr[end]-y_arr[end-u]) <= ϵ for u in 1:nϵ))
      x, y = x_best, y_best
    else
      break
    end
  end
  return x_best
end

## Corana Update Formula (Kochenderfer & Wheler Algorithm 8.5) ##
function corana_update!(v, a, c, ns)
  for i in 1:length(v)
    ai, ci = a[i], c[i]
    if ai > 0.6*ns
      v[i] *= (1 + ci*(ai/ns - 0.6)/0.4)
    elseif ai < 0.4*ns
      v[i] /= (1 + ci*(0.4-ai/ns)/0.4)
    end
  end
  return v
end