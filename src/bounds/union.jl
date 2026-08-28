# Unions of ellipsoids (or of cube-ellipsoid mixtures), and the machinery that decides when to
# split one bound into two.
#
# Port of nautilus/bounds/union.py. The Python class is called `Union`; that name is taken in
# Julia, hence `UnionBound`.

"""
    golden_section_min(f, a, b; tol=1e-8, max_iter=200) -> (x, f(x))

Minimum of a unimodal `f` on `[a, b]`.

Stands in for the one call to `scipy.optimize.minimize` in the package (union.py:39), which
uses L-BFGS-B from `x0 = 0.5` with numerical gradients. The objective there is smooth with a
single interior minimum, so a golden-section search finds the same point without a dependency
and without relying on a finite-difference gradient.
"""
function golden_section_min(f, a::Float64, b::Float64; tol::Float64 = 1e-8, max_iter::Int = 200)
    invphi = (sqrt(5.0) - 1) / 2
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in 1:max_iter
        b - a < tol && break
        if fc < fd
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
        end
    end
    x = (a + b) / 2
    return x, f(x)
end

"""
    ellipsoids_overlap(ellipsoids) -> Bool

Whether any two of `ellipsoids` overlap, by the criterion of ieeexplore.ieee.org/document/6289830.

For each pair, `k(s) = 1 - dᵀ (A₁⁻¹/(1-s) + A₂⁻¹/s)⁻¹ d` equals 1 at both ends of `(0, 1)` and
dips in between; the pair overlaps when that dip stays above zero. Port of union.py:14.

`A⁻¹` is recovered as `B Bᵀ` rather than by inverting `A`, since `B` is already its Cholesky
factor — same matrix, better conditioned, one less inversion.
"""
function ellipsoids_overlap(ellipsoids::AbstractVector{<:Ellipsoid})
    n = length(ellipsoids)
    n < 2 && return false
    A_inv = [e.B * e.B' for e in ellipsoids]
    for i in 1:(n - 1), j in (i + 1):n
        d = ellipsoids[i].c .- ellipsoids[j].c
        # The matrix is a positive combination of two SPD matrices, so it is SPD and a
        # Cholesky solve is both correct and far cheaper than the pivoted QR that `\` on a
        # plain Matrix falls back to -- and this runs ~40 times per pair, per split.
        k = function (s)
            M = A_inv[i] ./ (1 - s) .+ A_inv[j] ./ s
            F = cholesky(Symmetric(M), check = false)
            return 1 - dot(d, issuccess(F) ? F \ d : Symmetric(M) \ d)
        end
        _, fmin = golden_section_min(k, 1e-9, 1 - 1e-9)
        fmin > 0 && return true
    end
    return false
end

# ---------------------------------------------------------------------------------------------

"""
    UnionBound{B}

A union of bounds of type `B` (`Ellipsoid` or `UnitCubeEllipsoidMixture`), optionally
intersected with the unit cube.

`block[i]` marks a bound that must not be split again — either because it holds too few points
to divide, or because splitting it was tried and made the total volume worse.
"""
mutable struct UnionBound{B}
    n_dim::Int
    enlarge_per_dim::Float64
    n_points_min::Int
    cube::Union{UnitCube,Nothing}
    points_bounds::Vector{Matrix{Float64}}
    bounds::Vector{B}
    log_v_all::Vector{Float64}
    block::Vector{Bool}
    points::Matrix{Float64}                 # cache of accepted samples not yet handed out
    n_sample::Int
    n_reject::Int
    rng::AbstractRNG
end

"""
    UnionBound(points; enlarge_per_dim=1.1, n_points_min=nothing, unit=true,
               bound_class=Ellipsoid, rng=Random.default_rng())

Start from a single bound around all of `points`. Port of `Union.compute` (union.py:81).
"""
function UnionBound(points::AbstractMatrix{<:Real}; enlarge_per_dim::Real = 1.1,
                    n_points_min::Union{Nothing,Integer} = nothing, unit::Bool = true,
                    bound_class::Type = Ellipsoid, rng::AbstractRNG = Random.default_rng())
    n_dim, n_points = size(points)
    npm = n_points_min === nothing ? n_dim + 1 : Int(n_points_min)
    npm < n_dim + 1 && throw(ArgumentError(
        "the number of points per bound ($npm) must exceed the number of dimensions ($n_dim)"))

    pts = Matrix{Float64}(points)
    b0 = bound_class(pts; enlarge_per_dim, rng)
    return UnionBound{bound_class}(
        n_dim, Float64(enlarge_per_dim), npm,
        unit ? UnitCube(n_dim; rng) : nothing,
        [pts], [b0], [log_volume(b0)], [n_points < 2 * npm],
        zeros(n_dim, 0), 0, 0, rng)
end

nbounds(u::UnionBound) = length(u.bounds)

function reset!(u::UnionBound, rng::Union{Nothing,AbstractRNG} = nothing)
    u.points = zeros(u.n_dim, 0)
    u.n_sample = 0
    u.n_reject = 0
    if rng !== nothing
        u.rng = rng
        u.cube !== nothing && reset!(u.cube, rng)
        for b in u.bounds
            reset!(b, rng)
        end
    end
    return u
end

"""
    split!(u; allow_overlap=true) -> Bool

Split the largest splittable bound in two, using a 2-component Gaussian mixture in that
bound's own frame. Returns whether any split happened. Port of `Union.split` (union.py:155).

The split is rejected, and the bound blocked, if the two children together enclose more volume
than the parent; the call then retries on the next-largest bound.
"""
function split!(u::UnionBound{B}; allow_overlap::Bool = true) where {B}
    # `B <: Ellipsoid` is known at compile time, so for a mixture union the whole
    # no-overlap path below folds away -- which is what makes `ellipsoids_overlap`, which
    # reads fields only `Ellipsoid` has, unreachable rather than merely unreached.
    if !allow_overlap && !(B <: Ellipsoid)
        throw(ArgumentError("'allow_overlap' can only be false if the bounds are ellipsoids."))
    end
    all(u.block) && return false

    index = argmax(ifelse.(u.block, -Inf, u.log_v_all))
    points = u.points_bounds[index]
    points_t = transform(u.bounds[index], points)

    gmm = fit_gaussian_mixture(points_t, 2; rng = u.rng, n_init = 10)
    p = component_log_prob(gmm, points_t)
    labels = [argmax(view(p, :, i)) for i in axes(p, 2)]

    # If either cluster came out too small, move the most compatible points across. The Python
    # gets the count from `np.bincount`, which returns a length-1 array when one label is
    # unused and so silently passes the check before failing downstream; counting both labels
    # explicitly, as here, routes that case into the same rebalancing.
    counts = [count(==(l), labels) for l in 1:2]
    if !all(counts .>= u.n_points_min)
        label = argmin(counts)
        labels[partialsortperm(view(p, label, :), 1:u.n_points_min, rev = true)] .= label
    end

    new_bounds = [B(points[:, labels .== l]; enlarge_per_dim = u.enlarge_per_dim, rng = u.rng)
                  for l in 1:2]

    if B <: Ellipsoid && !allow_overlap && ellipsoids_overlap(
            vcat(u.bounds[1:(index - 1)], u.bounds[(index + 1):end], new_bounds))
        return false
    end

    if logsumexp(log_volume.(new_bounds)) > log_volume(u.bounds[index])
        u.block[index] = true
        return split!(u; allow_overlap)
    end

    deleteat!(u.points_bounds, index)
    deleteat!(u.bounds, index)
    deleteat!(u.block, index)
    for l in 1:2
        push!(u.points_bounds, points[:, labels .== l])
    end
    append!(u.bounds, new_bounds)
    u.log_v_all = log_volume.(u.bounds)
    for l in 1:2
        push!(u.block, size(u.points_bounds[end - 2 + l], 2) < 2 * u.n_points_min)
    end

    reset!(u)
    return true
end

"""
    trim!(u; threshold=1e3) -> Bool

Drop the sparsest bound, if its point density is at least `threshold` times below the median of
the others. Port of `Union.trim` (union.py:233).

Unlike the Python, this also drops the bound's `block` entry. The Python leaves `block` one
element longer than `bounds`, which would break a later `split`; nautilus never splits after
trimming, so the inconsistency is latent there. Removing it costs nothing and is one fewer
thing to trip over.
"""
function trim!(u::UnionBound; threshold::Real = 1e3)
    nbounds(u) == 1 && return false

    log_r = [log(size(p, 2)) for p in u.points_bounds] .- log_volume.(u.bounds)
    index = argmin(log_r)
    others = log_r[setdiff(eachindex(log_r), index)]

    log_r[index] - median(others) < -log(threshold) || return false

    deleteat!(u.points_bounds, index)
    deleteat!(u.bounds, index)
    deleteat!(u.block, index)
    u.log_v_all = log_volume.(u.bounds)
    reset!(u)
    return true
end

function in_bound(u::UnionBound, points::AbstractMatrix)
    res = falses(size(points, 2))
    for b in u.bounds
        res .|= in_bound(b, points)
    end
    u.cube !== nothing && (res .&= in_bound(u.cube, points))
    return res
end

"""
    sample(u, n_points=100)

Draw from the union without double-counting the overlaps: a point falling inside `m` of the
bounds is kept with probability `1/m`. Port of `Union.sample` (union.py:293).

Sampling is done in blocks of 1000 and cached, so `n_sample`/`n_reject` accumulate an estimate
of the union's true volume as a side effect — which is what [`log_volume`](@ref) reports.
"""
function sample(u::UnionBound, n_points::Int = 100)
    while size(u.points, 2) < n_points
        n_sample = 1000
        p = exp.(u.log_v_all .- logsumexp(u.log_v_all))
        n_per_bound = rand(u.rng, Multinomial(n_sample, p))

        points = reduce(hcat, (sample(b, n) for (b, n) in zip(u.bounds, n_per_bound)
                               if n > 0); init = zeros(u.n_dim, 0))
        u.cube !== nothing && (points = points[:, in_bound(u.cube, points)])
        points = points[:, randperm(u.rng, size(points, 2))]

        n_bound = zeros(Int, size(points, 2))
        for b in u.bounds
            n_bound .+= in_bound(b, points)
        end
        keep = rand(u.rng, size(points, 2)) .> (1 .- 1 ./ n_bound)
        points = points[:, keep]

        u.points = hcat(u.points, points)
        u.n_sample += n_sample
        u.n_reject += n_sample - size(points, 2)
    end

    out = u.points[:, 1:n_points]
    u.points = u.points[:, (n_points + 1):end]
    return out
end

"""
    log_volume(u::UnionBound)

Log of the union's volume: the sum of the individual volumes, corrected by the Monte-Carlo
rejection rate for overlaps. Triggers a sampling round on first call, as the Python's lazy
`log_v` property does (union.py:331).
"""
function log_volume(u::UnionBound)
    u.n_sample == 0 && sample(u)
    return logsumexp(u.log_v_all) + log(1 - u.n_reject / u.n_sample)
end
