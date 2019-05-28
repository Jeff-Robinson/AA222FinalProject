## Generalized Pattern Search (Kochenderfer & Wheeler Algorithm 7.6) ##
function generalized_pattern_search(f, x, α, D, max_n_evals; γ=0.5)
  y = f(x)
  while true
    improved = false
    for (i, d) in enumerate(D)
      xp = x + α.*d
      yp = f(xp)
      if yp < y
        x, y, improved = xp, yp, true
        D = pushfirst!(deleteat!(D, i), d)
        if NUM_FXN_EVALS >= max_n_evals # termination
          return x
        end
        break
      end
    end
    if !improved
      α *= γ
    end
    if NUM_FXN_EVALS >= max_n_evals # termination
      return x
    end
  end
  return x
end