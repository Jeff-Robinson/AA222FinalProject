# Piecewise Damping Coeffcient Using Hyperbolic Tangent Fitting
# Jeff Robinson - jbrobin@stanford.edu

"""

### Syntax
    sm_pw_damp_coeff(bRH, bRL, bCL, bCH, ẏcritR, ẏcritC, ẏ)

### Description
Calculates the suspension damping coefficient for a piecewise damping coefficient including low- and high-speed rebound and compression damping. Uses hyperbolic tangent functions to approximate step functions between piecewise components.

### Arguments
  bRH - High Speed Rebound Damping Coefficient [kg/s]
  bRL - Low Speed Rebound Damping Coefficient [kg/s]
  bCL - Low Speed Compression Damping Coefficient [kg/s]
  bCH - High Speed Compression Damping Coefficient [kg/s]
  ẏcritR - High/Low Speed Rebound Damping Crossover [m/s]
  ẏcritC - High/Low Speed Compression Damping Crossover [m/s]
  ẏ - Current Damper Rate [m/s]

"""
function sm_pw_damp_coeff(
  bRH, bRL, 
  bCL, bCH, 
  ẏcritR, 
  ẏcritC,
  ẏ
  )
  steepness = 100.0
  A1 = (bCL - bCH)/2.0
  r1 = ẏcritC
  c1 = bCH + A1
  f1 = A1*tanh(steepness/abs(A1) * (ẏ-r1)) + c1

  A2 = (bRL - bCL)/2.0
  r2 = 0.0
  c2 = A2
  f2 = A2*tanh(steepness/abs(A2) * (ẏ-r2)) + c2

  A3 = (bRH - bRL)/2.0
  r3 = ẏcritR
  c3 = A3
  f3 = A3*tanh(steepness/abs(A3) * (ẏ-r3)) + c3
  pw_damp_coeff =  f1 + f2 + f3
  return pw_damp_coeff
end