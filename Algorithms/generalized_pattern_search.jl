## Generalized Pattern Search (Kochenderfer & Wheeler Algorithm 7.6) ##
function generalized_pattern_search(f, x, α, D, max_n_evals; γ=0.5)
  y = f(x)
  x_log = [x]
  evals_log = [0]
  while true
    improved = false
    for (i, d) in enumerate(D)
      xp = x + α.*d
      yp = f(xp)
      if yp < y
        x, y, improved = xp, yp, true
        D = pushfirst!(deleteat!(D, i), d)
        push!(x_log, x)
        push!(evals_log, NUM_FXN_EVALS)
        if NUM_FXN_EVALS >= max_n_evals # termination
          return x, x_log, evals_log
        end
        break
      end
      if NUM_FXN_EVALS >= max_n_evals # termination
        return x, x_log, evals_log
      end
    end
    if !improved
      α *= γ
    end
  end
end