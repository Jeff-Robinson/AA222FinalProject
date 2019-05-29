## Bracket Minimum (Kochenderfer & Wheeler Algorithm 3.1) ##
function bracket_minimum(f, x; s=1e-2, k=2.0)
  a, ya = x, f(x)
  b, yb = a + s, f(a + s)
  n_evals = 2
  if yb > ya
    a, b = b, a
    ya, yb = yb, ya
    s = -s
  end
  while true
    c, yc = b + s, f(b + s)
    n_evals += 1
    if yc > yb
      return a < c ? (a, c, n_evals) : (c, a, n_evals)
    end
    a, ya, b, yb = b, yb, c, yc
    s *= k
  end
end

## Fibonacci Search (Kochenderfer & Wheeler Algorithm 3.2) ##
function fibonacci_search(f, a, b, n; ϵ=0.01)
  s = (1-sqrt(5))/(1+sqrt(5))
  ρ = 1 / (MathConstants.golden*(1 - s^(n+1))/(1-s^n))
  d = ρ*b + (1-ρ)*a
  yd = f(d)
  for i in 1:n-1
    if i == n-1
      c = ϵ*a + (1-ϵ)*d
    else
      c = ρ*a + (1-ρ)*b
    end
    yc = f(c)
    if yc < yd
      b, d, yd = d, c, yc
    else
      a, b = b, c
    end
    ρ = 1 / (MathConstants.golden*(1-s^(n-i+1))/(1-s^(n-i)))
  end
  # return a < b ? (a, b) : (b, a)
  return (a+b)/2.0
end

## Line Search (Kochenderfer & Wheeler Algorithm 4.1) ##
function line_search(f, x, max_n_evals)
  a, b, bracket_n_evals = bracket_minimum(f, x)
  x_best = fibonacci_search(f, a, b, max_n_evals - bracket_n_evals)
  return x_best
end
