# A multi-layer perceptron with Adam, reproducing `sklearn.neural_network.MLPRegressor` closely
# enough to stand in for it.
#
# nautilus asks for exactly one configuration (neural.py:83-85): hidden layers (100, 50, 20),
# `alpha=0`, `learning_rate_init=1e-2`, `max_iter=10000`, `tol=0`, `n_iter_no_change=10`. Every
# other setting is a scikit-learn default, and those defaults are the specification here --
# ReLU hidden layers with an identity output, Adam with the usual betas, Glorot-uniform init,
# batches of at most 200 reshuffled each epoch, and a stopping rule driven by the training loss.
#
# Pulling in a neural-network framework for a network this small would cost more in dependency
# weight and per-epoch overhead than it saves; this is a few hundred lines and exact.

"""
    MLPRegressor

Dense feed-forward network. `W[l]` is `(n_out, n_in)` — the transpose of scikit-learn's
`coefs_[l]`, which is the natural orientation for column-major samples.
"""
struct MLPRegressor
    W::Vector{Matrix{Float64}}
    b::Vector{Vector{Float64}}
end

"""
    MLPRegressor(n_in, hidden, n_out, rng)

Glorot-uniform initialisation: each weight and bias is drawn from `U(-c, c)` with
`c = sqrt(6 / (fan_in + fan_out))`. That is scikit-learn's `_init_coef` for a ReLU network,
biases included — which is not the more common "biases start at zero".
"""
function MLPRegressor(n_in::Int, hidden::NTuple{N,Int}, n_out::Int, rng::AbstractRNG) where {N}
    sizes = (n_in, hidden..., n_out)
    W = Matrix{Float64}[]
    b = Vector{Float64}[]
    for l in 1:(length(sizes) - 1)
        fan_in, fan_out = sizes[l], sizes[l + 1]
        c = sqrt(6.0 / (fan_in + fan_out))
        push!(W, (rand(rng, fan_out, fan_in) .* 2 .- 1) .* c)
        push!(b, (rand(rng, fan_out) .* 2 .- 1) .* c)
    end
    return MLPRegressor(W, b)
end

nlayers(m::MLPRegressor) = length(m.W)

"""
    predict(m::MLPRegressor, X) -> Vector

Forward pass over `X`, one sample per column. ReLU on the hidden layers, identity on the
output, as scikit-learn uses for regression.
"""
function predict(m::MLPRegressor, X::AbstractMatrix{Float64})
    L = nlayers(m)
    A = X
    for l in 1:L
        A = m.W[l] * A .+ m.b[l]
        l < L && (A = max.(A, 0.0))
    end
    return vec(A)
end

# Forward pass keeping the activations, then backward. Returns the batch loss and the
# gradients. Mirrors sklearn's `_backprop`; with `alpha = 0` the L2 terms drop out entirely.
function _backprop(m::MLPRegressor, X::AbstractMatrix{Float64}, y::AbstractVector{Float64})
    L = nlayers(m)
    n = size(X, 2)

    A = Vector{Matrix{Float64}}(undef, L + 1)
    A[1] = X
    for l in 1:L
        Z = m.W[l] * A[l] .+ m.b[l]
        l < L && (Z .= max.(Z, 0.0))
        A[l + 1] = Z
    end

    # sklearn's squared_loss is ((y - yhat)^2).mean() / 2, i.e. halved MSE.
    delta = A[L + 1] .- y'
    loss = sum(abs2, delta) / (2n)

    gW = Vector{Matrix{Float64}}(undef, L)
    gb = Vector{Vector{Float64}}(undef, L)
    for l in L:-1:1
        gW[l] = (delta * A[l]') ./ n
        gb[l] = vec(sum(delta, dims = 2)) ./ n
        if l > 1
            # ReLU derivative, keyed on the post-activation being non-zero exactly as
            # sklearn's `inplace_relu_derivative` does.
            delta = (m.W[l]' * delta) .* (A[l] .> 0)
        end
    end
    return loss, gW, gb
end

# sklearn's `gen_batches`: whole batches followed by whatever remains.
function _batch_ranges(n::Int, bs::Int)
    r = UnitRange{Int}[]
    start = 1
    for _ in 1:(n ÷ bs)
        push!(r, start:(start + bs - 1))
        start += bs
    end
    start <= n && push!(r, start:n)
    return r
end

"""
    fit!(m::MLPRegressor, X, y; rng, kwargs...) -> m

Train in place with Adam. `X` is `(n_features, n_samples)`.

The stopping rule is scikit-learn's, and its details matter: the count is driven by the
*training* loss (there is no validation split, since `early_stopping` defaults to false), an
epoch that fails to beat the running best increments it, any epoch that ties or beats it resets
it to zero, and training stops once the count is **strictly greater** than `n_iter_no_change` —
so eleven consecutive non-improvements, not ten. With `tol = 0` "improvement" means strictly
lower.
"""
function fit!(m::MLPRegressor, X::AbstractMatrix{Float64}, y::AbstractVector{Float64};
              rng::AbstractRNG, learning_rate_init::Float64 = 1e-2, max_iter::Int = 10_000,
              tol::Float64 = 0.0, n_iter_no_change::Int = 10, beta_1::Float64 = 0.9,
              beta_2::Float64 = 0.999, epsilon::Float64 = 1e-8, batch_size::Int = 200)
    n = size(X, 2)
    bs = clamp(batch_size, 1, n)

    # Adam state, kept in two concretely-typed groups rather than one heterogeneous list.
    # Collecting weights and biases into a single `Any[]` costs roughly 3x in run time here,
    # because every update then dispatches dynamically.
    L = nlayers(m)
    mW = [zeros(size(W)) for W in m.W]
    vW = [zeros(size(W)) for W in m.W]
    mb = [zeros(size(b)) for b in m.b]
    vb = [zeros(size(b)) for b in m.b]
    t = 0

    best_loss = Inf
    no_improvement = 0
    idx = collect(1:n)
    batches = _batch_ranges(n, bs)

    for _ in 1:max_iter
        shuffle!(rng, idx)
        accumulated = 0.0

        for s in batches
            cols = view(idx, s)
            loss, gW, gb = _backprop(m, X[:, cols], y[cols])
            accumulated += loss * length(s)

            t += 1
            # sklearn folds Adam's bias correction into the step size rather than into m and v.
            lr = learning_rate_init * sqrt(1 - beta_2^t) / (1 - beta_1^t)
            for l in 1:L
                @. mW[l] = beta_1 * mW[l] + (1 - beta_1) * gW[l]
                @. vW[l] = beta_2 * vW[l] + (1 - beta_2) * gW[l]^2
                @. m.W[l] -= lr * mW[l] / (sqrt(vW[l]) + epsilon)
                @. mb[l] = beta_1 * mb[l] + (1 - beta_1) * gb[l]
                @. vb[l] = beta_2 * vb[l] + (1 - beta_2) * gb[l]^2
                @. m.b[l] -= lr * mb[l] / (sqrt(vb[l]) + epsilon)
            end
        end

        loss = accumulated / n
        no_improvement = loss > best_loss - tol ? no_improvement + 1 : 0
        loss < best_loss && (best_loss = loss)
        no_improvement > n_iter_no_change && break
    end
    return m
end
