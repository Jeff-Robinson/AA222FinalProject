# Physical Simulation of Bicycle Tire on Rough Surface
# Jeff Robinson - jbrobin@stanford.edu

using Distributions
using DifferentialEquations
using PyPlot

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
  ramp_strength = 0.1
  # k1 = k1_0 * ( ramp_strength*(-log10(10*(x1 + (1-sag_point))) - log10(10*((sag_point) - x1)) + 1.39794000867203772) + 1) #2*log10(5) + 1 )
  k1 = k1_0 * ( exp(-10*(x1 + (1-sag_point))) + exp(10*(x1-sag_point)) + 1)
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
  k2 = k2_0 * ( 0.5*(tanh(10*(1 - x2)) + 1) + exp(-10*x2) )
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
tire_thk = 0.0635 # m, 2.50 inch 650B tire https://www.bikecalc.com/wheel_size_math
tire_OD = 0.711 # m, 6.50 inch 650B tire, characteristic length scale for trail surface noise
susp_travel = 0.2 # m
sag_point = 0.0 # portion of suspension travel used with no load applied
init_state = [0.0, #y0_0 [m]
              tire_thk, # y2_0 [m]
              tire_thk + (1-sag_point)*susp_travel, # y1_0 [m]
              0.0, # y2_dot_0 [m/s]
              0.0  # y1_dot_0 [m/s]
]
tspan = (0.0,30.0)
params = [50.0, # m1 [kg]
          5.0, # m2 [kg]
          tire_thk + (1-sag_point)*susp_travel, # y1_0 [m]
          tire_thk, # y2_0 [m]
          6000.0, # k1_0 [N/m]
          20000.0, # k2_0 [N/m]
          1000.0, # bRH [N-s/m]
          1000.0, # bRL [N-s/m]
          1000.0, # bCL [N-s/m]
          1000.0, # bCH [N-s/m]
          2.0, # y_dotcritR [m/s]
          -2.0 # y_dotcritC [m/s]
]
dtime = 0.0001 #sec
x_vel = 10.0 # [m/s], horizontal velocity
char_pt_scale = Integer(round(tire_OD/x_vel/dtime)) # number of time steps required to traverse one characteristic length scale (tire OD)

# ODE PROBLEM DEFINITION
times = []
moving_avg_incline = zeros(char_pt_scale)
accels = []
function suspension_model(dy, y, p, t)
  # y = [y0, y2, y1, y2_dot, y1_dot]
  # p = [m1, m2, y1_0, y2_0, k1_0, k2_0, bRH, bRL, bCL, bCH, y_dotcritR, y_dotcritC]
  push!(times, t)
  m1, m2, y1_0, y2_0, k1_0, k2_0, bRH, bRL, bCL, bCH, y_dotcritR, y_dotcritC = p
  travel_zero_point = y1_0 - y2_0
  x1 = ((y[3]-y[2]) - travel_zero_point)/susp_travel
  k1 = k_1(k1_0,x1)
  x2 = (y[2]-y[1])/y2_0
  k2 = k_2(k2_0,x2)
  b1 = sm_pw_damp_coeff(bRH, bRL, bCL, bCH, y_dotcritR, y_dotcritC, y[5]-y[4])

  pushfirst!(moving_avg_incline, rand(Normal(0.0,50.0)))
  dy[1] = sum(moving_avg_incline[1:char_pt_scale])/char_pt_scale
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
# @timev # Timer macro
Suspension_Sim_Sols = solve(Suspension_Sim_Prob,Euler();dt=dtime) # Requires 0.0001 sec time step
# Alternate Solvers Tested when encountering stiffness issues
# Suspension_Sim_Sols = solve(Suspension_Sim_Prob,AutoTsit5(Rosenbrock23()))
# Suspension_Sim_Sols = solve(Suspension_Sim_Prob,Rodas5())
# Suspension_Sim_Sols = solve(Suspension_Sim_Prob,Rosenbrock23())

# OBJECTIVE CRITERIA
ground_following = sum((Suspension_Sim_Sols[2,:].-Suspension_Sim_Sols[1,:])/tire_thk)/length(Suspension_Sim_Sols) - 1
# jerks = [(accels[i]-accels[i-1])/dtime for i=2:length(accels)]
ride_comfort = sqrt(sum((accels.+g).^2)/length(accels)) #+ 0.01*sqrt(sum(jerks.^2)/length(jerks))
println("Ground Following (mean distance from tire to ground): ",ground_following, "\nRide Comfort (RMS vertical acceleration of bike frame): ", ride_comfort)



# PLOTTING
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
ax5.plot(times, -x1, color = (0.6,0.6,0.6), linestyle = "-")
ax5.plot(times, -x2, color = (0.8,0.8,0.8), linestyle = "-")
ax5.set_title("Normalized Suspension Travel/Tire Displacement")
ax5.set_xlabel("Time, [s] / Spring Constant Multiplier, [nondim.]")
ax5.set_ylabel("Normalized Displacement")
ax5.plot(k1s, -xs, color = (0,0,0), linestyle = "-")
ax5.plot(k2s, -xs, color = (0,0,0), linestyle = "--")
ax5.set_xlim((0,times[end]))
ax5.set_ylim((-2,1.5))
ax5.legend(("x1","x2","k1/k1_0","k2/k2_0"))
