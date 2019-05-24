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
    xr[end-1] = abs(xr[end-1])
    xr[end] = -abs(xr[end])
    yr = f(xr)

    num_evals += 1 # end optimization if max number of evals reached
    println(num_evals, " / ", num_evals_max)
    if num_evals >= num_evals_max
        return S[argmin(y_arr)]
    end

    if yr < yl
        xe = xm + β * (xr - xm) # expansion point
        xe[end-1] = abs(xe[end-1])
        xe[end] = -abs(xe[end])
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
        xc[end-1] = abs(xc[end-1])
        xc[end] = -abs(xc[end])
        yc = f(xc)

        num_evals += 1 # end optimization if max number of evals reached
        println(num_evals, " / ", num_evals_max)
        if num_evals >= num_evals_max
            return S[argmin(y_arr)]
        end

        if yc > yh
            for i in 2:length(y_arr)
                S[i] = (S[i] + xl)/2
                S[i][end-1] = abs(S[i][end-1])
                S[i][end] = -abs(S[i][end])
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