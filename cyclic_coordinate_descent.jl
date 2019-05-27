## Bracket Minimum (Kochenderfer & Wheeler Algorithm 3.1) ##
function bracket_minimum(f, x, n_evals_used; s=1e-2, k=2.0)
  a = x
  ya, n_evals = f(x, n_evals_used)
  b = a + s
  yb, n_evals = f(a + s, n_evals)
  if yb > ya
    a, b = b, a
    ya, yb = yb, ya
    s = -s
  end
  while true
    c = b + s
    yc, n_evals = f(b + s, n_evals)
    if yc > yb
      return a < c ? (a, c, n_evals) : (c, a, n_evals)
    end
    a, ya, b, yb = b, yb, c, yc
    s *= k
  end
end

## Fibonacci Search (Kochenderfer & Wheeler Algorithm 3.2) ##
function fibonacci_search(f, a, b, max_n_evals, n_evals_used; ϵ=0.01)
  s = (1-sqrt(5))/(1+sqrt(5))
  ρ = 1 / (MathConstants.golden*(1 - s^(max_n_evals+1))/(1-s^max_n_evals))
  d = ρ*b + (1-ρ)*a
  yd, n_evals = f(d, n_evals_used)
  for i in 1:max_n_evals-1
    if i == max_n_evals-1
      c = ϵ*a + (1-ϵ)*d
    else
      c = ρ*a + (1-ρ)*b
    end
    yc, n_evals = f(c, n_evals)
    if yc < yd
      b, d, yd = d, c, yc
    else
      a, b = b, c
    end
    ρ = 1 / (MathConstants.golden*(1-s^(max_n_evals-i+1))/(1-s^(max_n_evals-i)))
  end
  # return a < b ? (a, b) : (b, a)
  return (a+b)/2.0, n_evals
end

## Line Search (Kochenderfer & Wheeler Algorithm 4.1) ##
function line_search(f, x, max_n_evals, n_evals_used)
  a, b, n_evals_used = bracket_minimum(f, x, n_evals_used)
  x_best, n_evals = fibonacci_search(f, a, b, max_n_evals, n_evals_used)
  return x_best, n_evals
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
      f_comp = (x_comp, n_evals) -> f([j == i ? x_comp : x[j] for j=1:n_dims], n_evals)
      x_comp_best, n_evals = line_search(f_comp, x[i], evals_per_search, n_evals)
      x = [j == i ? x_comp_best : x[j] for j=1:n_dims]
      push!(x_log, x)
    end
    Δ = norm(x - xp)
  end
  return x, x_log
end