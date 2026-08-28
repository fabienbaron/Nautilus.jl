# The composite bound: a union of cube-ellipsoid mixtures for sampling, intersected with a
# union of neural bounds for accuracy.
#
# Port of nautilus/bounds/nautilus.py.
#
# Two separate unions are built over the same points, and the distinction is the heart of the
# method. The INNER union is of plain ellipsoids, split as far as it can go without any two
# overlapping, which makes each ellipsoid a clean, separable mode; one NeuralBound is trained
# per ellipsoid. The OUTER union is of cube-ellipsoid mixtures and is what points are actually
# drawn from, split and trimmed only until its volume is within `split_threshold` of the
# target. Sampling is therefore cheap and the acceptance test is sharp.
#
# The `periodic` / PhaseShift handling of the Python is not ported. It would enter here, as a
# transform applied to `points` on the way in and inverted on the way out of `sample`.

mutable struct NautilusBound
    n_dim::Int
    neural_bounds::Vector{NeuralBound}
    outer_bound::UnionBound{UnitCubeEllipsoidMixture}
    rng::AbstractRNG
    points::Matrix{Float64}
    n_sample::Int
    n_reject::Int
end

"""
    NautilusBound(points, log_l, log_l_min, log_v_target; kwargs...)

Port of `NautilusBound.compute` (nautilus.py:41).

`log_v_target` is the volume the bound is aiming for — the live set's volume, in the sampler.
The outer union is split while it exceeds that target by more than
`split_threshold * enlarge_per_dim^n_dim`, then, if it is still too large, sparse ellipsoids
are trimmed away under the same criterion.
"""
function NautilusBound(points::AbstractMatrix{<:Real}, log_l::AbstractVector{<:Real},
                       log_l_min::Real, log_v_target::Real;
                       enlarge_per_dim::Real = 1.1,
                       n_points_min::Union{Nothing,Integer} = nothing,
                       split_threshold::Real = 100, n_networks::Int = 4,
                       rng::AbstractRNG = Random.default_rng(), kwargs...)
    n_dim = size(points, 1)
    pts = Matrix{Float64}(points)
    ll = Vector{Float64}(log_l)
    above = ll .>= log_l_min

    # Inner union: ellipsoids that must not overlap, one neural bound each.
    inner = UnionBound(pts[:, above]; enlarge_per_dim, n_points_min,
                       bound_class = Ellipsoid, rng)
    while split!(inner; allow_overlap = false)
    end

    neural_bounds = NeuralBound[]
    for ell in inner.bounds
        sel = in_bound(ell, pts)
        push!(neural_bounds,
              NeuralBound(pts[:, sel], ll[sel], log_l_min;
                          enlarge_per_dim, n_networks, rng, kwargs...))
    end

    # Outer union: what gets sampled from.
    outer = UnionBound(pts[:, above]; enlarge_per_dim, n_points_min,
                       bound_class = UnitCubeEllipsoidMixture, rng)

    threshold = log(split_threshold * enlarge_per_dim^n_dim)
    while log_volume(outer) - log_v_target > threshold
        split!(outer) || break
    end
    while log_volume(outer) - log_v_target > threshold
        trim!(outer) || break
    end

    return NautilusBound(n_dim, neural_bounds, outer, rng, zeros(n_dim, 0), 0, 0)
end

function in_bound(b::NautilusBound, points::AbstractMatrix)
    res = in_bound(b.outer_bound, points)
    if any(res)
        # Only points already inside the outer bound reach the networks: evaluating the
        # emulator is the expensive half, so the cheap test goes first.
        sub = points[:, res]
        inner = falses(size(sub, 2))
        for nb in b.neural_bounds
            inner .|= in_bound(nb, sub)
        end
        res[res] .= inner
    end
    return res
end

"""
    sample(b::NautilusBound, n_points=100; return_points=true)

Draw from the outer union and keep what any neural bound accepts, caching the surplus.
`return_points=false` fills the cache without handing anything back, which is how the sampler
primes the volume estimate.
"""
function sample(b::NautilusBound, n_points::Int = 100; return_points::Bool = true,
                threaded::Bool = false)
    if threaded && Threads.nthreads() > 1 && size(b.points, 2) < n_points
        sample_threaded!(b, n_points)
    end
    while size(b.points, 2) < n_points
        n_sample = 1000
        pts = sample(b.outer_bound, n_sample)
        keep = falses(size(pts, 2))
        for nb in b.neural_bounds
            keep .|= in_bound(nb, pts)
        end
        pts = pts[:, keep]
        b.points = hcat(b.points, pts)
        b.n_sample += n_sample
        b.n_reject += n_sample - size(pts, 2)
    end

    return_points || return nothing
    out = b.points[:, 1:n_points]
    b.points = b.points[:, (n_points + 1):end]
    return out
end

"""
    sample_threaded!(b, n_points)

Fill the point cache using every available thread, then merge.

The Python parallelises this by pickling the whole bound to worker processes, each with an
RNG spawned from the parent's (nautilus.py:228-241). Threads need no pickling, but the outer
union carries a mutable sample cache and reject counters, so each task gets its own copy of it
and the counters are summed at the end. The neural bounds are only ever read, so they are
shared.

Results depend on the thread count, since that fixes how the work is partitioned. A run is
reproducible for a given seed and a given `Threads.nthreads()`.
"""
function sample_threaded!(b::NautilusBound, n_points::Int)
    n_jobs = Threads.nthreads()
    per_job = (max(n_points - size(b.points, 2), 10_000) ÷ n_jobs) + 1
    seeds = rand(b.rng, UInt64, n_jobs)
    results = Vector{Tuple{Matrix{Float64},Int,Int,Int,Int}}(undef, n_jobs)

    Threads.@threads for j in 1:n_jobs
        outer = deepcopy(b.outer_bound)
        reset!(outer, Xoshiro(seeds[j]))
        chunks = Matrix{Float64}[]
        got, n_sample, n_reject = 0, 0, 0
        while got < per_job
            pts = sample(outer, 1000)
            keep = falses(size(pts, 2))
            for nb in b.neural_bounds
                keep .|= in_bound(nb, pts)
            end
            acc = pts[:, keep]
            push!(chunks, acc)
            got += size(acc, 2)
            n_sample += 1000
            n_reject += 1000 - size(acc, 2)
        end
        results[j] = (reduce(hcat, chunks; init = zeros(b.n_dim, 0)),
                      n_sample, n_reject, outer.n_sample, outer.n_reject)
    end

    for (pts, n_sample, n_reject, o_sample, o_reject) in results
        b.points = hcat(b.points, pts)
        b.n_sample += n_sample
        b.n_reject += n_reject
        b.outer_bound.n_sample += o_sample
        b.outer_bound.n_reject += o_reject
    end
    return b
end

function log_volume(b::NautilusBound)
    b.n_sample == 0 && sample(b; return_points = false)
    return log_volume(b.outer_bound) + log(1 - b.n_reject / b.n_sample)
end

"Number of outer bounds that actually use an ellipsoid, as reported in the status line."
n_ell(b::NautilusBound) = count(x -> any(!, x.dim_cube), b.outer_bound.bounds)

"Total number of trained networks across all neural bounds."
function n_net(b::NautilusBound)
    isempty(b.neural_bounds) && return 0
    e = b.neural_bounds[1].emulator
    return e === nothing ? 0 : length(b.neural_bounds) * length(e.networks)
end

function reset!(b::NautilusBound, rng::Union{Nothing,AbstractRNG} = nothing)
    b.points = zeros(b.n_dim, 0)
    b.n_sample = 0
    b.n_reject = 0
    if rng !== nothing
        b.rng = rng
        reset!(b.outer_bound, rng)
    end
    return b
end
