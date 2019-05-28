## Bracket Minimum (Kochenderfer & Wheeler Algorithm 3.1) ##
function bracket_minimum(f, x; s=0.1, k=2.0)
  a = x
  ya = f(x)
  b = a + s
  yb = f(a + s)
  n_evals = 2  
  if yb > ya
    a, b = b, a
    ya, yb = yb, ya
    s = -s
  end
  while true
    c = b + s
    yc = f(b + s)
    n_evals += 1
    if yc >= yb
      return a < c ? (a, c, n_evals) : (c, a, n_evals)
    end
    a, ya, b, yb = b, yb, c, yc
    s *= k
  end
end

## Fibonacci Search (Kochenderfer & Wheeler Algorithm 3.2) ##
function fibonacci_search(f, a, b, max_n_evals, n_evals_used; ϵ=0.01)
  max_n_evals = max_n_evals - n_evals_used
  s = (1-sqrt(5))/(1+sqrt(5))
  ρ = 1 / (MathConstants.golden*(1 - s^(max_n_evals+1))/(1-s^max_n_evals))
  d = ρ*b + (1-ρ)*a
  yd = f(d)
  x_best = d
  for i in 1:max_n_evals-1
    if i == max_n_evals-1
      c = ϵ*a + (1-ϵ)*d
    else
      c = ρ*a + (1-ρ)*b
    end
    yc = f(c)
    if yc < yd
      x_best = c
      b, d, yd = d, c, yc
    else
      a, b = b, c
    end
    ρ = 1 / (MathConstants.golden*(1-s^(max_n_evals-i+1))/(1-s^(max_n_evals-i)))
  end
  # return a < b ? (a, b) : (b, a)
  return x_best
end

## Line Search (Kochenderfer & Wheeler Algorithm 4.1) ##
function line_search(f, x, max_n_evals)
  a, b, n_evals_used = bracket_minimum(f, x)
  x_best = fibonacci_search(f, a, b, max_n_evals, n_evals_used)
  return x_best
end

## Basis Vector (Kochenderfer & Wheeler Algorithm 7.1) ##
basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1:n]

## Cyclic Coordinate Descent (Kochenderfer & Wheeler Algorithm 7.2) ##
function cyclic_coordinate_descent(f, x, max_n_evals; evals_per_search)
  Δ, n_dims, n_evals = Inf, length(x), 0
  k_max = floor(Int64, max_n_evals/(n_dims*evals_per_search))
  x_log = [x]
  # while abs(Δ) > ϵ
  for k = 1:k_max
    xp = copy(x)
    for i in 1:n_dims
      d = basis(i, n_dims)
      f_comp = x_comp -> f([j == i ? x_comp : x[j] for j=1:n_dims])
      x_comp_best = line_search(f_comp, x[i], evals_per_search)
      x = [j == i ? x_comp_best : x[j] for j=1:n_dims]
      push!(x_log, x)
    end
    Δ = norm(x - xp)
  end
  return x, x_log
end