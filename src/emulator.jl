# The likelihood emulator: an ensemble of MLPs whose predictions are averaged.
#
# Port of nautilus/neural.py. One detail of the original is easy to miss and is preserved
# here: the per-network seeds are literally 0, 1, ..., n_networks-1 for every bound in the
# whole run (neural.py:97-99), so network initialisation is decoupled from the sampler's RNG
# and the only source of diversity within an ensemble is the seed. A `random_state` supplied
# by the caller is ignored, with a warning, for the same reason.

struct NeuralNetworkEmulator
    mean::Vector{Float64}
    scale::Vector{Float64}
    networks::Vector{MLPRegressor}
end

"""
    train_emulator(x, y; n_networks=4, hidden=(100, 50, 20), rng=nothing, kwargs...)

Train an ensemble on `x` (one point per column) against targets `y`.

Inputs are standardised per dimension first. `std` is taken with `corrected = false` to match
numpy's `np.std`, whose default is the population standard deviation where Julia's default is
the sample one.

The ensemble members are independent, so they are trained on separate threads with BLAS pinned
to one thread apiece — the direct analogue of the `threadpool_limits(limits=1)` wrapper the
Python puts around each `fit` (neural.py:11).
"""
function train_emulator(x::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                        n_networks::Int = 4, hidden::NTuple{N,Int} = (100, 50, 20),
                        kwargs...) where {N}
    xf = Matrix{Float64}(x)
    yf = Vector{Float64}(y)

    mu = vec(mean(xf, dims = 2))
    sc = vec(std(xf, dims = 2, corrected = false))
    xn = (xf .- mu) ./ sc

    networks = Vector{MLPRegressor}(undef, n_networks)
    n_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads for i in 1:n_networks
            rng = Xoshiro(i - 1)                     # seeds 0 .. n_networks-1, as in Python
            net = MLPRegressor(size(xn, 1), hidden, 1, rng)
            fit!(net, xn, yf; rng, kwargs...)
            networks[i] = net
        end
    finally
        BLAS.set_num_threads(n_blas)
    end

    return NeuralNetworkEmulator(mu, sc, networks)
end

"""
    predict(e::NeuralNetworkEmulator, x) -> Vector

Arithmetic mean of the ensemble's predictions for `x` (one point per column).
"""
function predict(e::NeuralNetworkEmulator, x::AbstractMatrix{<:Real})
    xn = (Matrix{Float64}(x) .- e.mean) ./ e.scale
    out = zeros(size(xn, 2))
    for net in e.networks
        out .+= predict(net, xn)
    end
    return out ./ length(e.networks)
end
