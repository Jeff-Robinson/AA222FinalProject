## Nelder-Mead Simplex Method (Kochenderfer & Wheeler Algorithm 7.7) ##
function nelder_mead(f, S, num_evals_max; α=1.0, β=2.0, γ=0.5)
s_length = length(S)
y_arr = Array{Float64,1}(undef, s_length)
num_evals = 0
for i = 1:s_length
    y_arr[i], num_evals = f(S[i], num_evals) # function values at simplex vertices
end

# num_evals = length(y_arr) # initialize function eval counter after init function values found

while true
    p = sortperm(y_arr) # sort lowest to highest
    S, y_arr = S[p], y_arr[p]
    xl, yl = S[1], y_arr[1] # lowest
    xh, yh = S[end], y_arr[end] # highest
    xs, ys = S[end-1], y_arr[end-1] # second-highest
    xm = mean(S[1:end-1]) # centroid
    xr = xm + α * (xm - xh) # reflection point
    xr = abs.(xr)
    xr[end] = -abs(xr[end])
    yr, num_evals = f(xr, num_evals)

    if num_evals >= num_evals_max # termination
        return S[argmin(y_arr)]
    end

    if yr < yl
        xe = xm + β * (xr - xm) # expansion point
        xe = abs.(xe)
        xe[end] = -abs(xe[end])
        ye, num_evals = f(xe, num_evals)

        if num_evals >= num_evals_max # termination
            return S[argmin(y_arr)]
        end

        S[end], y_arr[end] = ye < yr ? (xe, ye) : (xr, yr)
    elseif yr > ys
        if yr <= yh
            xh, yh, S[end], y_arr[end] = xr, yr, xr, yr
        end
        xc = xm + γ * (xh - xm) # contraction point
        xc = abs.(xc)
        xc[end] = -abs(xc[end])
        yc, num_evals = f(xc, num_evals)
        
        if num_evals >= num_evals_max # termination
            return S[argmin(y_arr)]
        end

        if yc > yh
            for i in 2:length(y_arr)
                S[i] = (S[i] + xl)/2
                S[i] = abs.(S[i])
                S[i][end] = -abs(S[i][end])
                y_arr[i], num_evals = f(S[i], num_evals)
                
                if num_evals >= num_evals_max # termination
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