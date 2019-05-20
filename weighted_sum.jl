# Weighted Sum Method
# Jeff Robinson - jbrobin@stanford.edu

function weighted_sum(
  f, 
  x, 
  weights
)
combined_f = sum(f(x).*weights)
return combined_f
end