# Randomized Trail Surface
# Jeff Robinson - jbrobin@stanford.edu

using Distributions

"""

### Syntax
    rand_trail_surface(time,vel,phys_spacing,sigma=0.25,moving_avg_dist=0.5)

### Description
Generates a randomized "trail surface" by sampling from a normal distribution and smoothing the result with a moving average

### Arguments
  time - length of time to simulate [s]
  vel - constant velocity of vehicle over ground [m/s]
  phys_spacing - physical spacing of points defining ground [m]
  sigma - standard deviation of ground surface [m]
  moving_avg_dist - physical distance to run moving average over to smooth ground surface [m]
  
"""
function rand_trail_surface(time,vel,phys_spacing,sigma=0.25,moving_avg_dist=0.5)
  dist = Normal(0.0,sigma)
  num_points = Integer(round(time * vel/phys_spacing))
  rand_trail_surface = rand(dist, num_points)
  moving_avg_count = Integer(round(moving_avg_dist/phys_spacing))
  println("moving average count: ",moving_avg_count, "total sample points: ",num_points)
  smooth_trail_surface = []
  for i = moving_avg_count+1:length(rand_trail_surface)
    push!(smooth_trail_surface, 
          sum(rand_trail_surface[i-moving_avg_count:i])/moving_avg_count
         )
  end
  return smooth_trail_surface
end