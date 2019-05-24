# Physical Simulation of Bicycle Tire on Rough Surface (Objective Function)
# Jeff Robinson - jbrobin@stanford.edu

using Distributions
# using DifferentialEquations
using OrdinaryDiffEq
using PyPlot
using Statistics

default_inputs = (state_vec = [6.0, 20.0, 1.0, 1.0, 1.0, 1.0, 2.0, -2.0],
tire_thk = 0.0635, # [m]
tire_OD = 0.711, # [m]
susp_travel = 0.2, # [m]
m1 = 50.0, # [kg]
m2 = 5.0, # [kg]
x_vel = 3.0, # [m/s], horizontal velocity
time_lim = 60.0, # [s], real time per Simulation
num_sims = 10, # number of iterations to average objectives over
plotting = false
)

"""

### Syntax
  `suspension_model_objective(state_vec::Array{Float64,1}; tire_thk::Float64, tire_OD::Float64, susp_travel::Float64, m1::Float64, m2::Float64, x_vel::Float64, time_lim::Float64, num_sims::Integer, plotting::Bool)`

  `suspension_model_objective(state_vec::Array{Float64,1})`

### Description
Runs a phsyical simulation of a suspended bicycle wheel traversing rough ground and returns averaged values for the "ground following" and "ride comfort" objectives. Ground following is defined as the mean distance between the tire and ground, normalized by the tire thickness to generate a value of O(1). Ride comfort is defined as the RMS vertical acceleration of the bicycle frame, subtracting gravity.

### Arguments
  `state_vec::Array{Float64,1}` - array of state variables\n
      k1_0, [kN/m], default: 6.0
      k2_0, [kN/m], default: 20.0
      bRH, [kN-s/m], default: 1.0
      bRL, [kN-s/m], default: 1.0
      bCL, [kN-s/m], default: 1.0
      bCH, [kN-s/m], default: 1.0
      y_dotcritR, [m/s], default: 2.0
      y_dotcritC, [m/s], default: -2.0

### Keyword Arguments
  `tire_thk::Float64` -> tire depth (travel), [m], default: 0.0635\n
  `tire_OD::Float64` -> tire diameter/ground length scale, [m], default: 0.711\n
  `susp_travel::Float64` -> suspension travel, [m], default: 0.2\n
  `m1::Float64` -> half of bicycle sprung mass (rider + frame), [kg], default: 50.0\n
  `m2::Float64` -> bicycle unsprung mass per wheel, [kg], default: 5.0\n
  `x_vel::Float64` -> horizontal velocity, [m/s], default: 3.0\n
  `time_lim::Float64` -> duration of simulation, [s], default: 30.0\n
  `num_sims::Integer` -> number of iterations to average objectives, default: 10\n
  `plotting::Bool` -> whether to generate plots from a single simulation run or generate averaged objective values over several simulations, default: false

"""
function suspension_model_objective(
  state_vec::Array{Float64,1} = default_inputs.state_vec;
  tire_thk::Float64 = default_inputs.tire_thk, # [m]
  tire_OD::Float64 = default_inputs.tire_OD, # [m]
  susp_travel::Float64 = default_inputs.susp_travel, # [m]
  m1::Float64 = default_inputs.m1, # [kg]
  m2::Float64 = default_inputs.m2, # [kg]
  x_vel::Float64 = default_inputs.x_vel, # [m/s], horizontal velocity
  time_lim::Float64 = default_inputs.time_lim, # [s], real time per Simulation
  num_sims::Integer = default_inputs.num_sims, # number of iterations to average objectives over
  plotting::Bool = default_inputs.plotting
)

state_orders = [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1.0, 1.0]

state_vec = state_vec.*state_orders

k1_0 = state_vec[1] # [N/m]
k2_0 = state_vec[2] # [N/m]
bRH = state_vec[3] # [N-s/m]
bRL = state_vec[4] # [N-s/m]
bCL = state_vec[5] # [N-s/m]
bCH = state_vec[6] # [N-s/m]
y_dotcritR = state_vec[7] # y_dotcritR [m/s]
y_dotcritC = state_vec[8] # y_dotcritC [m/s]

ramp = 5.0

"""

### Syntax
  k_2(k2_0, x2)

### Description
Calculates the spring constant of the tire as a function of tire displacement. Uses an exponential function to enforce the "bottom-out" condition when the tire uses its full travel. Uses a hyperbolic tangent function to approximate contact behavior, allowing for zero spring force when the tire is not in contact with the ground.

### Arguments
  k1_0 - Neutral spring constant of suspension [N/m]
  x2 - Normalized suspension travel, defined as the ratio between the suspension displacement from the zero-force "sag point" and the total possible suspension travel. x1 = ((y1-y2) - (y1_0-y2_0))/susp_travel

"""
function k_1(k1_0, x1)
  # ramp_strength = 0.1
  # k1 = k1_0 * ( ramp_strength*(-log10(10*(x1 + (1-sag_point))) - log10(10*((sag_point) - x1)) + 1.39794000867203772) + 1) #2*log10(5) + 1 )
  k1 = k1_0 * ( exp(-ramp*(x1 + (1-sag_point))) + exp(ramp*(x1-sag_point)) + 1)
  return k1
end

"""

### Syntax
  k_2(k2_0, x2)

### Description
Calculates the spring constant of the tire as a function of tire displacement. Uses an exponential function to enforce the "bottom-out" condition when the tire uses its full travel. Uses a hyperbolic tangent function to approximate contact behavior, allowing for zero spring force when the tire is not in contact with the ground.

### Arguments
  k2_0 - Neutral spring constant of tire [N/m]
  x2 - Normalized tire displacement, defined as the ratio of the distance between the "tire" and the "ground" and the initial, or zero-load tire displacement. x2 = (y2-y0)/y2_0

"""
function k_2(k2_0, x2)
  ramp_strength = 0.1
  # k2 = k2_0 * ( 0.5*(tanh(100*(1 - x2)) + 1) - ramp_strength*(log10(x2) + 0.2*x2) )
  k2 = k2_0 * ( 0.5*(tanh(100*(1 - x2)) + 1) + exp(-ramp*x2) )
  return k2
end

"""

### Syntax
  sm_pw_damp_coeff(bRH, bRL, bCL, bCH, ẏcritR, ẏcritC, ẏ)

### Description
Calculates the suspension damping coefficient for a four-regime piecewise damping coefficient including low- and high-speed rebound and compression damping. Uses hyperbolic tangent functions to create smooth steps between piecewise components.

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
  # ϵ = 1e-6
  A1 = (bCL - bCH)/2.0
  r1 = ẏcritC
  c1 = bCH + A1
  f1 = A1*tanh(steepness * (ẏ-r1)) + c1

  A2 = (bRL - bCL)/2.0
  r2 = 0.0
  c2 = A2
  f2 = A2*tanh(steepness * (ẏ-r2)) + c2

  A3 = (bRH - bRL)/2.0
  r3 = ẏcritR
  c3 = A3
  f3 = A3*tanh(steepness * (ẏ-r3)) + c3
  pw_damp_coeff =  f1 + f2 + f3
  return pw_damp_coeff
end

# INITIAL CONDITIONS
g = 9.8065 # [m/s^2]
# tire_thk = 0.0635 # m, 2.50 inch 650B tire https://www.bikecalc.com/wheel_size_math
# tire_OD = 0.711 # m, 6.50 inch 650B tire, characteristic length scale for trail surface noise
# susp_travel = 0.2 # m
sag_point = 0.0 # portion of suspension travel used with no load applied
init_state = [0.0, #y0_0 [m]
              tire_thk, # y2_0 [m]
              tire_thk + (1-sag_point)*susp_travel, # y1_0 [m]
              0.0, # y2_dot_0 [m/s]
              0.0  # y1_dot_0 [m/s]
]
tspan = (0.0, time_lim)
params = [
  m1, 
  m2, 
  init_state[3], # y1_0 [m]
  init_state[2], # y2_0 [m]
  k1_0,
  k2_0,
  bRH,
  bRL,
  bCL,
  bCH,
  y_dotcritR,
  y_dotcritC
]
dtime = 0.0001 #sec
# x_vel = 10.0 # [m/s], horizontal velocity
char_pt_scale = Integer(round(tire_OD/x_vel/dtime)) # number of time steps required to traverse one characteristic length scale (tire OD)
# char_time_scale = tire_OD/x_vel

# ODE PROBLEM DEFINITION
# times = []
moving_avg_incline = zeros(char_pt_scale+1)
accels = []
function suspension_model(dy, y, p, t)
  # y = [y0, y2, y1, y2_dot, y1_dot]
  # p = [m1, m2, y1_0, y2_0, k1_0, k2_0, bRH, bRL, bCL, bCH, y_dotcritR, y_dotcritC]
  # push!(times, t)
  m1, m2, y1_0, y2_0, k1_0, k2_0, bRH, bRL, bCL, bCH, y_dotcritR, y_dotcritC = p
  travel_zero_point = y1_0 - y2_0
  x1 = ((y[3]-y[2]) - travel_zero_point)/susp_travel
  k1 = k_1(k1_0, x1)
  x2 = (y[2]-y[1])/y2_0
  k2 = k_2(k2_0, x2)
  b1 = sm_pw_damp_coeff(bRH, bRL, bCL, bCH, y_dotcritR, y_dotcritC, y[5]-y[4])

  push!(moving_avg_incline, rand(Normal(-0.5,50.0)))
  dy[1] = mean(moving_avg_incline[end-char_pt_scale:end])
  dy[2] = y[4]
  dy[3] = y[5]
  dy[4] = -g + b1/m2*(y[5]-y[4]) + 
          k1/m2*((y[3]-y[2]) - travel_zero_point) - 
          k2/m2*((y[2]-y[1]) - y2_0)
  dy[5] = -g - b1/m1*(y[5]-y[4]) - 
          k1/m1*((y[3]-y[2]) - travel_zero_point)
  push!(accels, dy[5])
end

# SOLVING
Suspension_Sim_Prob = ODEProblem(suspension_model,init_state,tspan,params)

ground_following = []
ride_comfort = []
for i = 1:num_sims
  # Suspension_Sim_Sols = solve(Suspension_Sim_Prob,Euler();dt=dtime) # Requires 0.0001 sec time step
  # Suspension_Sim_Sols = solve(Suspension_Sim_Prob,Tsit5())
  Suspension_Sim_Sols = solve(Suspension_Sim_Prob,DP5())
  # OBJECTIVE CRITERIA
  if Suspension_Sim_Sols.t[end] == time_lim
    push!(ground_following, 
          # sqrt(sum(((Suspension_Sim_Sols[2,:].-Suspension_Sim_Sols[1,:])/tire_thk).^2)) )
          abs(sum((Suspension_Sim_Sols[2,:].-Suspension_Sim_Sols[1,:])/tire_thk)
          /length(Suspension_Sim_Sols) - 1))
    push!(ride_comfort, sqrt(sum((accels.+g).^2)/length(accels)))
  else
    push!(ground_following, Inf)
    push!(ride_comfort, Inf)
  end

end

if any(ground_following .== Inf) || any(ground_following .== NaN) || any(ride_comfort .== Inf) || any(ride_comfort .== NaN)
  ground_following_mean = Inf
  ride_comfort_mean = Inf
else
  ground_following_mean = sum(ground_following)/num_sims
  ride_comfort_mean = sum(ride_comfort)/num_sims
end

if plotting
  Suspension_Sim_Sols = solve(Suspension_Sim_Prob,Euler();dt=dtime) # Requires 0.0001 sec time step
  times = Suspension_Sim_Sols.t
  # ground_following = sum((Suspension_Sim_Sols[2,:].-Suspension_Sim_Sols[1,:])/tire_thk)/length(Suspension_Sim_Sols) - 1
  # ride_comfort = sqrt(sum((accels.+g).^2)/length(accels))

  travel_zero_point = (1-sag_point)*susp_travel
  x1 = ((Suspension_Sim_Sols[3,:] .- Suspension_Sim_Sols[2,:]) .- travel_zero_point)/susp_travel
  x2 = (Suspension_Sim_Sols[2,:] .- Suspension_Sim_Sols[1,:])./tire_thk

  fig1 = figure(0)
  # (ax1, ax2, ax3) = fig1.subplots(nrows=3, ncols=1)
  ax3 = fig1.subplots()
  # ax1.plot(times,Suspension_Sim_Sols[4,:])
  # ax1.set_title("Wheel Motion Rate, [m/s]")
  # ax2.plot(times,Suspension_Sim_Sols[5,:])
  # ax2.set_title("Bicycle Motion Rate, [m/s]")
  ax3.plot(times, Suspension_Sim_Sols[5,:]-Suspension_Sim_Sols[4,:],
          color = (0.4,0.4,0.4))
  Clim = ax3.plot(times, fill(params[end],length(times)), 
                  color = (0,0,0), linestyle = "--", label = "High/Low Speed Compression Crossover")
  Rlim = ax3.plot(times, fill(params[end-1],length(times)), 
                  color = (0,0,0), linestyle = ":", label = "High/Low Speed Rebound Crossover")
  ax3.set_title("Net Suspension Motion Rate, [m/s]")
  ax3.set_xlabel("Time, [s]")
  ax3.legend()

  fig2 = figure(1)
  ax4 = fig2.subplots()
  display_time = 2.5 # [s]
  display_pts = Integer(round(display_time/dtime))
  ax4.plot(times[1:display_pts]*x_vel,
          Suspension_Sim_Sols[1,1:display_pts], 
          color = (0.7,0.7,0.7))
  ax4.plot(times[1:display_pts]*x_vel,
          Suspension_Sim_Sols[2,1:display_pts], 
          color = (0.4,0.4,0.4))
  ax4.plot(times[1:display_pts]*x_vel,
          Suspension_Sim_Sols[3,1:display_pts],
          color = (0,0,0))
  ax4.set_title("Ground, Wheel, and Frame Position")
  ax4.set_xlabel("Distance, [m]")
  ax4.set_ylabel("Altitude, [m]")
  ax4.legend(("Ground","Wheel","Frame"))
  ax4.axis("equal")

  xs = [i for i in -1.5:0.01:2]
  k1s = [k_1(1.0, xval) for xval in xs ]
  k2s = [k_2(1.0, xval) for xval in xs ]
  # b1s = [sm_pw_damp_coeff(1000.0, # bRH [N-s/m]
  #                         1000.0, # bRL [N-s/m]
  #                         1000.0, # bCL [N-s/m]
  #                         1000.0, # bCH [N-s/m]
  #                         5.0, # y_dotcritR [m/s]
  #                         -5.0, # y_dotcritC [m/s]
  # Suspension_Sim_Sols[5,i]-Suspension_Sim_Sols[4,i]) 
  # for i=1:length(Suspension_Sim_Sols)]

  fig3 = figure(2)
  ax5 = fig3.subplots()
  ax5.plot(times, -x2, color = (0.8,0.8,0.8), linestyle = "-")
  ax5.plot(times, -x1, color = (0.6,0.6,0.6), linestyle = "-")
  ax5.set_title("Normalized Suspension Travel/Tire Displacement")
  ax5.set_xlabel("Time, [s] / Spring Constant Multiplier, [nondim.]")
  ax5.set_ylabel("Normalized Displacement")
  ax5.plot(k2s, -xs, color = (0,0,0), linestyle = "--")
  ax5.plot(k1s, -xs, color = (0,0,0), linestyle = "-")
  ax5.set_xlim((0,times[end]))
  ax5.set_ylim((-2,1.5))
  ax5.legend(("Tire Displacement","Suspension Travel","Tire Spring Constant Multiplier","Suspension Spring Constant Multiplier"))

end

return [ground_following_mean, ride_comfort_mean]

end