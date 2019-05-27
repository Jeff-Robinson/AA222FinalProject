## Generalized Pattern Search (Kochenderfer & Wheeler Algorithm 7.6) ##
function generalized_pattern_search(f, x, α, D, max_n_evals; γ=0.5)
  y, n_evals = f(x)
  while true
    improved = false
    for (i, d) in enumerate(D)
      xp = x + α.*d
      yp, n_evals = f(xp, n_evals)
      if yp < y
        x, y, improved = xp, yp, true
        D = pushfirst!(deleteat!(D, i), d)
        if n_evals >= max_n_evals # termination
          return x
        end
        break
      end
    end
    if !improved
      α *= γ
    end
    if n_evals >= max_n_evals # termination
      return x
    end
  end
  return x
end