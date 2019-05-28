## Nelder-Mead Simplex Method (Kochenderfer & Wheeler Algorithm 7.7) ##
function nelder_mead(f, S, max_n_evals; α=1.0, β=2.0, γ=0.5)
y_arr = f.(S) # function values at simplex vertices

while true
    p = sortperm(y_arr) # sort lowest to highest
    S, y_arr = S[p], y_arr[p]
    xl, yl = S[1], y_arr[1] # lowest
    xh, yh = S[end], y_arr[end] # highest
    xs, ys = S[end-1], y_arr[end-1] # second-highest
    xm = mean(S[1:end-1]) # centroid
    xr = xm + α * (xm - xh) # reflection point
    yr = f(xr)

    if NUM_FXN_EVALS >= max_n_evals # termination
        return S[argmin(y_arr)]
    end

    if yr < yl
        xe = xm + β * (xr - xm) # expansion point
        ye = f(xe)

        if NUM_FXN_EVALS >= max_n_evals # termination
            return S[argmin(y_arr)]
        end

        S[end], y_arr[end] = ye < yr ? (xe, ye) : (xr, yr)
    elseif yr > ys
        if yr <= yh
            xh, yh, S[end], y_arr[end] = xr, yr, xr, yr
        end
        xc = xm + γ * (xh - xm) # contraction point
        yc = f(xc)
        
        if NUM_FXN_EVALS >= max_n_evals # termination
            return S[argmin(y_arr)]
        end

        if yc > yh
            for i in 2:length(y_arr)
                S[i] = (S[i] + xl)/2
                y_arr[i] = f(S[i])
                
                if NUM_FXN_EVALS >= max_n_evals # termination
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