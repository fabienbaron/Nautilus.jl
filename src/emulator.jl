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

# BLAS has one thread count for the whole process, so pinning it to 1 around training is
# shared state. Saving and restoring it naively races when several samplers train at once:
# the second saves the 1 the first has just set, the first then restores the full count
# while the second is still training, and the second finally "restores" 1 and leaves the
# process there for good. So the pin is reference-counted: the first trainer in saves the
# count and pins, the last one out restores it.
const _BLAS_PIN_LOCK = ReentrantLock()
const _BLAS_PIN_COUNT = Ref(0)
const _BLAS_SAVED_THREADS = Ref(1)

function _with_single_blas_thread(f)
    lock(_BLAS_PIN_LOCK) do
        if _BLAS_PIN_COUNT[] == 0
            _BLAS_SAVED_THREADS[] = BLAS.get_num_threads()
            BLAS.set_num_threads(1)
        end
        _BLAS_PIN_COUNT[] += 1
    end
    try
        return f()
    finally
        lock(_BLAS_PIN_LOCK) do
            _BLAS_PIN_COUNT[] -= 1
            _BLAS_PIN_COUNT[] == 0 && BLAS.set_num_threads(_BLAS_SAVED_THREADS[])
        end
    end
end

"""
    train_emulator(x, y; n_networks=4, hidden=(100, 50, 20), threaded_training=true, kwargs...)

Train an ensemble on `x` (one point per column) against targets `y`.

Inputs are standardised per dimension first. `std` is taken with `corrected = false` to match
numpy's `np.std`, whose default is the population standard deviation where Julia's default is
the sample one.

The ensemble members are independent, so by default they are trained on separate threads,
with BLAS pinned to one thread for the duration. That is the analogue of the
`threadpool_limits(limits=1)` wrapper the Python puts around each `fit` (neural.py:11).
`threaded_training = false` trains them one after the other, for callers that already run
several samplers in parallel. Each network has a fixed seed, so the trained ensemble is
identical either way.
"""
function train_emulator(x::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                        n_networks::Int = 4, hidden::NTuple{N,Int} = (100, 50, 20),
                        threaded_training::Bool = true, kwargs...) where {N}
    xf = Matrix{Float64}(x)
    yf = Vector{Float64}(y)

    mu = vec(mean(xf, dims = 2))
    sc = vec(std(xf, dims = 2, corrected = false))
    xn = (xf .- mu) ./ sc

    networks = Vector{MLPRegressor}(undef, n_networks)
    function train(i)
        rng = Xoshiro(i - 1)                         # seeds 0 .. n_networks-1, as in Python
        net = MLPRegressor(size(xn, 1), hidden, 1, rng)
        fit!(net, xn, yf; rng, kwargs...)
        networks[i] = net
    end
    _with_single_blas_thread() do
        if threaded_training
            Threads.@threads for i in 1:n_networks
                train(i)
            end
        else
            foreach(train, 1:n_networks)
        end
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
