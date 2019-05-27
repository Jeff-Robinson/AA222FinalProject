using Random
using Distributions
using LinearAlgebra

## Covariance Matrix Adaptation ("Algorithms For Optimization" Algorithm 8.9, Kochenderfer & Wheeler) ##
function covariance_matrix_adaptation(f, x, max_n_evals;
    σ = 0.5,
    m = 4 + floor(Int, 3*log(length(x))),
    m_elite = div(m, 2),
    )
    k_max = floor(Int64, (max_n_evals - 1)/(m+1))
    x_best = x
    y_best, n_evals = f(x_best)

    μ, n_dims = copy(x), length(x)
    ws = normalize!(vcat(log((m+1)/2) .- log.(1:m_elite), 
                    zeros(m - m_elite)), 
                    1)
    μ_eff = 1 / sum(ws.^2)
    cσ = (μ_eff + 2)/(n_dims + μ_eff + 5)
    dσ = 1 + 2*max(0, sqrt((μ_eff-1)/(n_dims+1))-1) + cσ
    cΣ = (4 + μ_eff/n_dims)/(n_dims + 4 + 2*μ_eff/n_dims)
    c1 = 2/((n_dims+1.3)^2 + μ_eff)
    cμ = min(1-c1, 2*(μ_eff-2+1/μ_eff)/((n_dims+2)^2 + μ_eff))
    E = n_dims^0.5*(1-1/(4*n_dims)+1/(21*n_dims^2))
    pσ, pΣ, Σ = zeros(n_dims), zeros(n_dims), Matrix(1.0I, n_dims, n_dims)

    for k in 1:k_max
        P = MvNormal(μ, σ^2*Σ)
        xs = [rand(P) for i in 1:m]
        for i=1:m # ensure comp. and rebound crossovers are correct sign
          xs[i][end-1] = abs(xs[i][end-1])
          xs[i][end] = -abs(xs[i][end])
        end
        for x in xs
          ys, n_evals = f(x, n_evals)
        end
        
        is = sortperm(ys) # best to worst

        # selection and mean update
        δs = [(x - μ)/σ for x in xs]
        δw = sum(ws[i]*δs[is[i]] for i in 1:m_elite)
        μ += σ*δw

        # step size control
        C = Σ^(-0.5)
        pσ = (1-cσ)*pσ + sqrt(cσ*(2-cσ)*μ_eff)*C*δw
        σ *= exp(cσ/dσ * (norm(pσ)/E - 1))

        # covariance Adaptation
        hσ = Int(norm(pσ)/sqrt(1-(1-cσ)^(2*k)) < (1.4+2/(n_dims+1))*E)
        pΣ = (1-cΣ)*pΣ + hσ*sqrt(cΣ*(2-cΣ)*μ_eff)*δw
        w0 = [ws[i]>=0 ? ws[i] : n_dims*ws[i]/norm(C*δs[is[i]])^2 for i in 1:m]
        Σ = (1-c1-cμ) * Σ + c1*(pΣ*pΣ' + (1-hσ) * cΣ*(2-cΣ) * Σ) + cμ*sum(w0[i]*δs[is[i]]*δs[is[i]]' for i in 1:m)
        Σ = triu(Σ) + triu(Σ,1)' # enforce symmetry

        x_best_potential = μ
        y_best_potential, n_evals = f(μ, n_evals)
        if ys[is[1]] < y_best_potential
          x_best_potential = xs[is[1]]
          y_best_potential = ys[is[1]]
        end
        if y_best_potential < y_best
            x_best = x_best_potential
            y_best = y_best_potential
        end
    end
    return x_best
end
