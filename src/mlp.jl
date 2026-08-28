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
#
# MEASURED against SimpleChains.jl, the strongest candidate, on this exact workload
# (100 epochs, batch 200, one BLAS thread; BenchmarkTools minimum of 10 samples):
#
#     d     n    this   SC Float64   SC Float32
#     6  1000   53.3 ms    38.5 ms      25.3 ms
#     6  5000  283.5 ms   192.3 ms     129.7 ms
#     2  2000  110.4 ms    75.0 ms      50.8 ms
#    20  2000  124.2 ms    86.2 ms      57.4 ms
#
# So SimpleChains is 1.4x ahead in Float64 and 2.2x in Float32. Against that: it pulls in
# LoopVectorization, it encodes the input dimension in the TYPE so every new dimensionality
# pays fresh codegen (measured at 2.4 s for d=2, 9.7 s for d=6, 2.7 s for d=20, on top of an
# already-real TTFX), Float32 is where its best numbers are, and adopting it means
# reimplementing the init and the stopping rule around its training loop. The network is 68%
# of a sampler run, so its remaining Float64 margin is worth about 1.2x end to end. Not worth
# it -- but the gap was 2.0-3.8x before this file was made allocation-free, so the judgement
# rests on that and would flip back if this regressed.

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

"""
    MLPWorkspace

Scratch space for training, allocated once per `fit!` and reused by every batch of every
epoch.

The obvious implementation allocates the activations, both gradient arrays and a copy of the
batch on each of ~500 backprop calls per network, and at this size that allocation traffic is
a large fraction of the run time rather than a rounding error. `A[l]` and `D[l]` are sized for
the largest batch and used through views, so the short final batch costs nothing extra.
"""
struct MLPWorkspace
    A::Vector{Matrix{Float64}}     # A[l]: activations of layer l, (size_l, batch_max)
    D::Vector{Matrix{Float64}}     # D[l]: delta at layer l's output, (size_{l+1}, batch_max)
    gW::Vector{Matrix{Float64}}
    gb::Vector{Vector{Float64}}
    y::Vector{Float64}             # the batch's targets, gathered
end

function MLPWorkspace(m::MLPRegressor, batch_max::Int)
    L = nlayers(m)
    sizes = [size(m.W[1], 2); [size(W, 1) for W in m.W]]
    A = [Matrix{Float64}(undef, sizes[l], batch_max) for l in 1:(L + 1)]
    D = [Matrix{Float64}(undef, sizes[l + 1], batch_max) for l in 1:L]
    return MLPWorkspace(A, D, [similar(W) for W in m.W], [similar(b) for b in m.b],
                        Vector{Float64}(undef, batch_max))
end

# Gather one batch into the workspace. Indexing `X[:, cols]` would allocate a fresh matrix per
# batch; this writes into the buffer that is already there.
function _gather!(ws::MLPWorkspace, X::AbstractMatrix{Float64}, y::AbstractVector{Float64},
                  cols::AbstractVector{Int})
    d = size(X, 1)
    @inbounds for (k, c) in enumerate(cols)
        for j in 1:d
            ws.A[1][j, k] = X[j, c]
        end
        ws.y[k] = y[c]
    end
    return nothing
end

# Forward then backward over the batch already sitting in `ws`. Returns the batch loss; the
# gradients are left in `ws.gW` / `ws.gb`. Mirrors sklearn's `_backprop`; with `alpha = 0` the
# L2 terms drop out entirely.
function _backprop!(ws::MLPWorkspace, m::MLPRegressor, bs::Int)
    L = nlayers(m)

    for l in 1:L
        Al = view(ws.A[l], :, 1:bs)
        Z = view(ws.A[l + 1], :, 1:bs)
        mul!(Z, m.W[l], Al)
        bl = m.b[l]
        @inbounds for j in 1:bs, i in eachindex(bl)
            z = Z[i, j] + bl[i]
            Z[i, j] = (l < L && z < 0) ? 0.0 : z
        end
    end

    # sklearn's squared_loss is ((y - yhat)^2).mean() / 2, i.e. halved MSE.
    out = view(ws.A[L + 1], :, 1:bs)
    dL = view(ws.D[L], :, 1:bs)
    loss = 0.0
    @inbounds for j in 1:bs
        e = out[1, j] - ws.y[j]
        dL[1, j] = e
        loss += e * e
    end
    loss /= 2bs

    for l in L:-1:1
        Dl = view(ws.D[l], :, 1:bs)
        Al = view(ws.A[l], :, 1:bs)
        mul!(ws.gW[l], Dl, transpose(Al))
        ws.gW[l] ./= bs
        gb = ws.gb[l]
        @inbounds for i in eachindex(gb)
            acc = 0.0
            for j in 1:bs
                acc += Dl[i, j]
            end
            gb[i] = acc / bs
        end
        if l > 1
            Dp = view(ws.D[l - 1], :, 1:bs)
            mul!(Dp, transpose(m.W[l]), Dl)
            # ReLU derivative, keyed on the post-activation being non-zero exactly as
            # sklearn's `inplace_relu_derivative` does.
            @inbounds for j in 1:bs, i in axes(Dp, 1)
                Al[i, j] > 0 || (Dp[i, j] = 0.0)
            end
        end
    end
    return loss
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
    ws = MLPWorkspace(m, maximum(length, batches))

    for _ in 1:max_iter
        shuffle!(rng, idx)
        accumulated = 0.0

        for sl in batches
            cols = view(idx, sl)
            nb = length(sl)
            _gather!(ws, X, y, cols)
            loss = _backprop!(ws, m, nb)
            accumulated += loss * nb

            t += 1
            # sklearn folds Adam's bias correction into the step size rather than into m and v.
            lr = learning_rate_init * sqrt(1 - beta_2^t) / (1 - beta_1^t)
            for l in 1:L
                gW, gb = ws.gW[l], ws.gb[l]
                @. mW[l] = beta_1 * mW[l] + (1 - beta_1) * gW
                @. vW[l] = beta_2 * vW[l] + (1 - beta_2) * gW^2
                @. m.W[l] -= lr * mW[l] / (sqrt(vW[l]) + epsilon)
                @. mb[l] = beta_1 * mb[l] + (1 - beta_1) * gb
                @. vb[l] = beta_2 * vb[l] + (1 - beta_2) * gb^2
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
