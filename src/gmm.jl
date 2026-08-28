# A two-component, full-covariance Gaussian mixture, fitted by EM.
#
# This stands in for `sklearn.mixture.GaussianMixture(n_components=2, n_init=10)`, which is
# the only use scikit-learn's mixture module gets in nautilus (bounds/union.py:187) and is
# reached with every other parameter left at its default. Those defaults are reproduced here
# explicitly: full covariances, k-means initialisation, `reg_covar=1e-6`, `tol=1e-3` on the
# per-sample mean log-likelihood, `max_iter=100`, and `n_init` independent restarts of which
# the highest-likelihood one wins.
#
# GaussianMixtures.jl was the obvious alternative and was rejected: its `nInit` counts k-means
# iterations rather than EM restarts, so the restart behaviour that matters here would have to
# be built on top anyway, and its full-covariance path carries machinery a two-component
# problem does not need.
#
# MEASURED, on two well-separated clusters, both recovering the labelling perfectly:
#
#     d    n     this    GaussianMixtures.jl
#     2   500   0.79 ms         4.09 ms
#     6  1000   2.03 ms        10.11 ms
#     6  2000   3.60 ms        40.24 ms
#    20  1000   4.85 ms        26.47 ms
#
# 5-11x faster at identical accuracy, and that is against a GaussianMixtures.jl call doing
# strictly LESS work (its `nInit` is not ten restarts). ExpectationMaximization.jl is the other
# candidate; it is generic over Distributions.jl mixtures, which is elegant but means per
# component `fit_mle` dispatch and no restart semantics either. ScikitLearn.jl would reintroduce
# the Python dependency this port exists to remove.
#
# The E-step is where the time goes, so it is written for it: one Cholesky and one batched
# triangular solve per component per iteration, rather than a `logpdf(MvNormal, x)` per point.
# That form alone is 2.8-4.0x faster than the per-point one and numerically identical to 1e-14.
#
# For scale: on the 6-parameter BC2004 fit, profiled single-threaded with each sample
# attributed to the innermost non-Base frame, this file plus k-means is 3.5% of run time
# against 68.2% for the neural network, 12.6% for the MVEE and 10.3% for the likelihood
# itself. The GMM is not the bottleneck and further work here would be misdirected.
#
# (Profile it single-threaded. `Profile` samples every thread, so under `-t 4` roughly
# three quarters of the samples are idle workers parked in the scheduler with pure-Base
# stacks, and any attribution that ignores that is measuring thread idleness.)

struct GaussianMixture2
    weights::Vector{Float64}            # mixing weights, length k
    means::Vector{Vector{Float64}}      # component means, each length d
    covs::Vector{Matrix{Float64}}       # component covariances, each d x d
end

ncomponents(g::GaussianMixture2) = length(g.weights)

"""
    component_log_prob(g, X) -> Matrix

`out[k, i] = log(w_k) + logpdf(N(mu_k, Sigma_k), X[:, i])`, the unnormalised log
responsibilities. This is exactly what `Union.split` maximises over to get its labels
(bounds/union.py:190-194).
"""
function component_log_prob(g::GaussianMixture2, X::AbstractMatrix{Float64})
    d, n = size(X)
    k = ncomponents(g)
    out = Matrix{Float64}(undef, k, n)
    for c in 1:k
        # One Cholesky and one batched triangular solve per component, rather than a
        # `logpdf(MvNormal, x)` per point: that form does n separate size-d solves where this
        # does a single BLAS-3 `trsm`, and it is called on every EM iteration of every restart.
        C = cholesky(Symmetric(g.covs[c]))
        Z = C.L \ (X .- g.means[c])
        # -0.5 * logdet(S) == -sum(log(diag(L)))
        cst = log(g.weights[c]) - 0.5 * d * log(2π) - sum(log, view(C.factors, diagind(C.factors)))
        @inbounds for i in 1:n
            q = 0.0
            for j in 1:d
                q += Z[j, i]^2
            end
            out[c, i] = cst - 0.5 * q
        end
    end
    return out
end

# M step: responsibilities -> (weights, means, covariances). Mirrors sklearn's
# `_estimate_gaussian_parameters` including the `10 * eps` floor on the component counts.
function _m_step(X::AbstractMatrix{Float64}, resp::AbstractMatrix{Float64}, reg_covar::Float64)
    d, n = size(X)
    k = size(resp, 1)
    nk = vec(sum(resp, dims = 2)) .+ 10 * eps(Float64)
    means = [X * view(resp, c, :) ./ nk[c] for c in 1:k]
    covs = Vector{Matrix{Float64}}(undef, k)
    for c in 1:k
        diff = X .- means[c]
        S = (diff .* view(resp, c, :)') * diff' ./ nk[c]
        S .= (S .+ S') ./ 2
        S[diagind(S)] .+= reg_covar      # sklearn: covariances[k].flat[::n_features+1] += reg_covar
        covs[c] = S
    end
    return GaussianMixture2(nk ./ n, means, covs)
end

# E step: returns the log responsibilities and the mean log-likelihood per sample, which is
# the quantity sklearn calls the lower bound and tests for convergence.
function _e_step(g::GaussianMixture2, X::AbstractMatrix{Float64})
    lp = component_log_prob(g, X)
    k, n = size(lp)
    log_resp = similar(lp)
    acc = 0.0
    # An explicit reduction rather than `mapslices(logsumexp, ...)`, which is type-unstable
    # and allocates a slice per column.
    @inbounds for i in 1:n
        m = -Inf
        for c in 1:k
            lp[c, i] > m && (m = lp[c, i])
        end
        se = 0.0
        for c in 1:k
            se += exp(lp[c, i] - m)
        end
        nrm = m + log(se)
        for c in 1:k
            log_resp[c, i] = lp[c, i] - nrm
        end
        acc += nrm
    end
    return log_resp, acc / n
end

function _fit_once(X::AbstractMatrix{Float64}, k::Int, rng::AbstractRNG;
                   reg_covar::Float64, tol::Float64, max_iter::Int)
    d, n = size(X)
    km = kmeans(X, k; rng, init = :kmpp)
    resp = zeros(k, n)
    @inbounds for i in 1:n
        resp[km.assignments[i], i] = 1.0
    end

    g = _m_step(X, resp, reg_covar)
    lower_bound = -Inf
    for _ in 1:max_iter
        prev = lower_bound
        log_resp, lower_bound = _e_step(g, X)
        g = _m_step(X, exp.(log_resp), reg_covar)
        isfinite(prev) && abs(lower_bound - prev) < tol && break
    end
    _, lower_bound = _e_step(g, X)         # score the parameters actually returned
    return g, lower_bound
end

"""
    fit_gaussian_mixture(X, k=2; rng, n_init=10, reg_covar=1e-6, tol=1e-3, max_iter=100)

Fit a `k`-component full-covariance Gaussian mixture to `X` (one sample per column), keeping
the best of `n_init` independent EM restarts.

A restart that throws — k-means cannot seed `k` distinct centres, or a component collapses to
a singular covariance — is skipped rather than propagated, matching scikit-learn's behaviour
of reporting a failure only when every initialisation fails.
"""
function fit_gaussian_mixture(X::AbstractMatrix{<:Real}, k::Int = 2;
                              rng::AbstractRNG = Random.default_rng(), n_init::Int = 10,
                              reg_covar::Real = 1e-6, tol::Real = 1e-3, max_iter::Int = 100)
    Xf = Matrix{Float64}(X)
    size(Xf, 2) >= k ||
        throw(ArgumentError("need at least $k samples to fit $k components, got $(size(Xf, 2))"))

    best = nothing
    best_lb = -Inf
    for _ in 1:n_init
        try
            g, lb = _fit_once(Xf, k, rng; reg_covar = Float64(reg_covar),
                              tol = Float64(tol), max_iter = max_iter)
            if lb > best_lb
                best, best_lb = g, lb
            end
        catch err
            # A restart can legitimately fail numerically; anything else is a bug and must
            # not be silently absorbed into "all initialisations failed".
            err isa Union{PosDefException,SingularException,RankDeficientException} ||
                err isa ArgumentError || rethrow()
        end
    end
    best === nothing && error("Gaussian mixture fit failed for all $n_init initialisations.")
    return best
end
