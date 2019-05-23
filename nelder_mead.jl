## Nelder-Mead Simplex Method (Kochenderfer & Wheeler Algorithm 7.7) ##
function nelder_mead(f, S, num_evals_max; α=1.0, β=2.0, γ=0.5)
y_arr = f.(S) # function values at simplex vertices

num_evals = length(y_arr) # initialize function eval counter after init function values found

while true
    p = sortperm(y_arr) # sort lowest to highest
    S, y_arr = S[p], y_arr[p]
    xl, yl = S[1], y_arr[1] # lowest
    xh, yh = S[end], y_arr[end] # highest
    xs, ys = S[end-1], y_arr[end-1] # second-highest
    xm = mean(S[1:end-1]) # centroid
    xr = xm + α * (xm - xh) # reflection point
    yr = f(xr)

    num_evals += 1 # end optimization if max number of evals reached
    println(num_evals, " / ", num_evals_max)
    if num_evals >= num_evals_max
        return S[argmin(y_arr)]
    end

    if yr < yl
        xe = xm + β * (xr - xm) # expansion point
        ye = f(xe)

        num_evals += 1 # end optimization if max number of evals reached
        println(num_evals, " / ", num_evals_max)
        if num_evals >= num_evals_max
            return S[argmin(y_arr)]
        end

        S[end], y_arr[end] = ye < yr ? (xe, ye) : (xr, yr)
    elseif yr > ys
        if yr <= yh
            xh, yh, S[end], y_arr[end] = xr, yr, xr, yr
        end
        xc = xm + γ * (xh - xm) # contraction point
        yc = f(xc)

        num_evals += 1 # end optimization if max number of evals reached
        println(num_evals, " / ", num_evals_max)
        if num_evals >= num_evals_max
            return S[argmin(y_arr)]
        end

        if yc > yh
            for i in 2:length(y_arr)
                S[i] = (S[i] + xl)/2
                y_arr[i] = f(S[i])
                
                num_evals += 1 # end optimization if max number of evals reached
                println(num_evals, " / ", num_evals_max)
                if num_evals >= num_evals_max
                    return S[argmin(y_arr)]
                end
            end
        else
            S[end], y_arr[end] = xc, yc
        end
    else
        S[end], y_arr[end] = xr, yr
    end
end
end


# """

# Arguments:
# - `f`: Function to be optimized
# - `g`: Gradient function for `f`
# - `x0`: (Vector) Initial position to start from
# - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
# - `prob`: (String) Name of the problem. So you can use a different strategy for each problem
# """
# function optimize(f, g, x0, n, prob)
# S = [x0] # initialize simplex with given random point
# for i=1:length(x0) # fill out remaining simplex points by projecting from given random point to points on origin axes
#     if prob == "simple_2"
#         push!(S, clamp.(randn(length(x0)), -3.0, 3.0)) # randomized simplex works better for Simple 2 for some reason
#     else
#         zvec = zeros(length(x0))
#         zvec[i] = x0[i]
#         push!(S, zvec)
#     end
# end
# return nelder_mead(f, S, n)
# end