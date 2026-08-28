# The basic geometric bounds: the unit cube, the minimum-volume enclosing ellipsoid, and the
# per-dimension mixture of the two.
#
# Port of nautilus/bounds/basic.py. Points are stored as (n_dim, n_points) here, one point per
# COLUMN, where the Python stores them as (n_points, n_dim). That is the one systematic
# difference from the original and it is deliberate: every hot loop below walks one point at a
# time, and a column is contiguous in Julia where a row is not. It also matches what
# Clustering.jl and Distributions.jl expect, so nothing needs transposing between here and
# src/gmm.jl. The Python line references in the comments therefore describe the same
# operation with the axes swapped.

"""
    invert_spd(m) -> Matrix

Inverse of a symmetric positive-definite matrix, via its Cholesky factor.

The Python (`invert_symmetric_positive_semidefinite_matrix`, basic.py:154) calls LAPACK
`dpotrf`/`dpotri` by hand and then symmetrises the returned triangle. `inv(cholesky(...))`
reaches those same two routines and hands back the full matrix, so the symmetrisation step
has no counterpart here.
"""
function invert_spd(m::AbstractMatrix)
    F = cholesky(Symmetric(m), check = false)
    issuccess(F) && return inv(F)
    # A degenerate point cloud -- everything on a line or a plane -- makes this matrix
    # singular. The Python takes `dpotrf(m)[0]` and never looks at the `info` flag LAPACK
    # returns beside it, so it carries on with an unfactorised triangle; the pseudo-inverse
    # is the same decision made deliberately.
    return pinv(Symmetric(m))
end

# Lower Cholesky factor, tolerating a spectrum that has touched zero.
function _cholesky_L(A::AbstractMatrix)
    F = cholesky(Symmetric(A), check = false)
    issuccess(F) && return Matrix(F.L)
    E = eigen(Symmetric(A))
    lo = eps(Float64) * max(maximum(E.values), one(Float64))
    return Matrix(cholesky(Symmetric(E.vectors * Diagonal(max.(E.values, lo)) * E.vectors')).L)
end

# g[i] = q_iᵀ M q_i for every column of q, written into g.
function _quadform!(g::Vector{Float64}, q::AbstractMatrix{Float64}, M::AbstractMatrix{Float64})
    Mq = M * q
    @inbounds for i in eachindex(g)
        s = 0.0
        for j in axes(q, 1)
            s += q[j, i] * Mq[j, i]
        end
        g[i] = s
    end
    return g
end

"""
    minimum_volume_enclosing_ellipsoid(points; n_max=100, n_batch=20) -> (c, A, A_inv)

Approximate the minimum-volume enclosing ellipsoid `(x - c)ᵀ A (x - c) ≤ 1` with a batched
variant of the Khachiyan algorithm.

Port of `minimum_volume_enclosing_ellipsoid` (basic.py:175). Two details of the original are
load-bearing and reproduced literally rather than tidied up:

  * The Python refreshes `v`/`v_inv` from scratch when `i % 1000 == 0`, which with `n_max=100`
    means only on the very first iteration; every later update is incremental. Hoisting that
    out of the loop, as done here, is what the code actually does.
  * Within a batch, the first candidate reuses the `g` computed for the whole batch while
    every later one is recomputed against the just-updated `v_inv`. In the Python that falls
    out of rebinding `g` from an array to a scalar and catching the resulting `IndexError`
    (basic.py:222-225); here it is the `cached` flag. It changes the result, so it stays.
"""
function minimum_volume_enclosing_ellipsoid(points::AbstractMatrix{Float64};
                                            n_max::Int = 100, n_batch::Int = 20)
    n_dim, n_points = size(points)
    m = n_dim + 1                                    # working dimension, q is (d+1) x n
    q = vcat(points, ones(1, n_points))
    u = fill(1.0 / n_points, n_points)

    # q_outer[:, :, i] = q_i q_iᵀ, precomputed for every point as the Python does. This is the
    # memory bottleneck of the routine: O(n (d+1)^2).
    q_outer = Array{Float64}(undef, m, m, n_points)
    @inbounds for i in 1:n_points
        for k in 1:m, j in 1:m
            q_outer[j, k, i] = q[j, i] * q[k, i]
        end
    end

    v = q * Diagonal(u) * q'
    v_inv = invert_spd(v)
    g = Vector{Float64}(undef, n_points)
    nb = min(n_batch, n_points)

    for _ in 1:n_max
        _quadform!(g, q, v_inv)
        cached = true
        for j in partialsortperm(g, 1:nb, rev = true)     # the nb largest, descending
            gj = if cached
                g[j]
            else
                s = 0.0
                @inbounds for k in 1:m, l in 1:m
                    s += q_outer[l, k, j] * v_inv[l, k]
                end
                s
            end
            cached = false
            gj < m && continue
            a = (gj - m) / (m * (gj - 1))
            @inbounds for k in 1:m, l in 1:m
                v[l, k] = v[l, k] * (1 - a) + a * q_outer[l, k, j]
            end
            v_inv = invert_spd(v)
            u .*= (1 - a)
            u[j] += a
        end
    end

    su = sum(u)
    c = (points * u) ./ su                            # np.average(points, weights=u)
    X = points .- c
    A_inv = ((X .* u') * X') ./ su                    # np.cov(..., aweights=u, bias=True)
    A_inv .= (A_inv .+ A_inv') ./ 2                   # kill the asymmetry the product leaves
    A = inv(A_inv)
    A .= (A .+ A') ./ 2

    # Grow the ellipsoid just enough to enclose every point.
    scale = maximum(sum(X .* (A * X), dims = 1))
    A ./= scale
    A_inv .*= scale

    return c, A, A_inv
end

# ---------------------------------------------------------------------------------------------
# UnitCube
# ---------------------------------------------------------------------------------------------

mutable struct UnitCube
    n_dim::Int
    rng::AbstractRNG
end

UnitCube(n_dim::Integer; rng::AbstractRNG = Random.default_rng()) = UnitCube(Int(n_dim), rng)

in_bound(b::UnitCube, points::AbstractMatrix) =
    vec(all((points .>= 0) .& (points .< 1), dims = 1))

sample(b::UnitCube, n_points::Int = 100) = rand(b.rng, b.n_dim, n_points)

log_volume(::UnitCube) = 0.0

reset!(b::UnitCube, rng::AbstractRNG) = (b.rng = rng; b)

# ---------------------------------------------------------------------------------------------
# Ellipsoid
# ---------------------------------------------------------------------------------------------

"""
    Ellipsoid

The bound `(x - c)ᵀ A (x - c) ≤ 1`. `B` is the Cholesky factor of `A⁻¹` and `B_inv` its
inverse, so that `transform` whitens the ellipsoid to the unit ball.
"""
mutable struct Ellipsoid
    n_dim::Int
    c::Vector{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    B_inv::Matrix{Float64}
    rng::AbstractRNG
end

"""
    Ellipsoid(points; enlarge_per_dim=1.1, rng=Random.default_rng())

Minimum-volume enclosing ellipsoid of `points` (one point per column), enlarged by
`enlarge_per_dim` along each semi-axis.

Port of `Ellipsoid.compute` (basic.py:266). The enlargement is applied *after* the MVEE has
been rescaled to enclose every point, so the enclosing guarantee survives it and the volume
grows by exactly `enlarge_per_dim^n_dim`.
"""
function Ellipsoid(points::AbstractMatrix{<:Real}; enlarge_per_dim::Real = 1.1,
                   rng::AbstractRNG = Random.default_rng())
    n_dim, n_points = size(points)
    enlarge_per_dim < 1 &&
        throw(ArgumentError("'enlarge_per_dim' cannot be smaller than unity, got $enlarge_per_dim"))
    n_points > n_dim ||
        throw(ArgumentError("number of points ($n_points) must exceed number of dimensions ($n_dim)"))

    c, A, A_inv = minimum_volume_enclosing_ellipsoid(Matrix{Float64}(points))
    A ./= enlarge_per_dim^2
    A_inv .*= enlarge_per_dim^2

    B = _cholesky_L(A_inv)                            # np.linalg.cholesky is lower-triangular
    return Ellipsoid(n_dim, c, A, B, inv(B), rng)
end

"Whiten `points` into the ellipsoid's frame, where the bound is the unit ball."
transform(e::Ellipsoid, points::AbstractMatrix) = e.B_inv * (points .- e.c)

"Undo [`transform`](@ref)."
transform_inverse(e::Ellipsoid, points::AbstractMatrix) = e.B * points .+ e.c

in_bound(e::Ellipsoid, points::AbstractMatrix) =
    vec(sum(abs2, transform(e, points), dims = 1)) .< 1

function sample(e::Ellipsoid, n_points::Int = 100)
    d = e.n_dim
    pts = randn(e.rng, d, n_points)
    r = rand(e.rng, n_points)
    @inbounds for i in 1:n_points
        col = @view pts[:, i]
        col ./= norm(col)                             # onto the unit sphere ...
        col .*= r[i]^(1 / d)                          # ... then in, uniformly by volume
    end
    return transform_inverse(e, pts)
end

# log of the ellipsoid volume: log|B| plus the volume of the d-ball. Written in the Python's
# form (basic.py:384) rather than the simplified (d/2)log(pi) - lgamma(d/2+1) for traceability.
log_volume(e::Ellipsoid) =
    logabsdet(e.B)[1] + e.n_dim * log(2) + e.n_dim * loggamma(1.5) - loggamma(e.n_dim / 2 + 1)

reset!(e::Ellipsoid, rng::AbstractRNG) = (e.rng = rng; e)

# ---------------------------------------------------------------------------------------------
# UnitCubeEllipsoidMixture
# ---------------------------------------------------------------------------------------------

"""
    UnitCubeEllipsoidMixture

Bounds each dimension either by the unit interval or by an ellipsoid over the remaining
dimensions, whichever gives the smaller volume. `dim_cube[i]` is true where dimension `i` is
handled by the cube.
"""
mutable struct UnitCubeEllipsoidMixture
    n_dim::Int
    dim_cube::BitVector
    cube::Union{UnitCube,Nothing}
    ellipsoid::Union{Ellipsoid,Nothing}
    rng::AbstractRNG
end

"""
    UnitCubeEllipsoidMixture(points; enlarge_per_dim=1.1, rng=Random.default_rng())

Port of `UnitCubeEllipsoidMixture.compute` (basic.py:472), a two-phase greedy search.

Phase one starts from a full-dimensional ellipsoid and moves dimensions to the cube one at a
time, each time choosing the dimension whose projection promises the largest volume drop, and
stops as soon as a move fails to reduce the volume. That is greedy and can land above the unit
cube itself, so phase two restarts from the all-cube configuration and adds ellipsoid
dimensions back, re-opening every previously-cube dimension whenever one is accepted.

The Python tracks phase two's incumbent with `UnitCube.compute(points)` — passing the point
array where the constructor wants `n_dim`. It survives only because the result is used solely
for its `log_v`, which is the constant `0`. Here that incumbent is the number `0.0` outright.
"""
function UnitCubeEllipsoidMixture(points::AbstractMatrix{<:Real}; enlarge_per_dim::Real = 1.1,
                                  rng::AbstractRNG = Random.default_rng())
    n_dim = size(points, 1)
    kw = (; enlarge_per_dim, rng)

    # -- phase one: peel dimensions off the ellipsoid ------------------------------------------
    ellipsoid = Ellipsoid(points; kw...)
    dim_cube = falses(n_dim)

    while count(!, dim_cube) > 1
        c = ellipsoid.c
        A_inv = inv(ellipsoid.A)
        free = findall(!, dim_cube)
        P = points[free, :]

        # Project the ellipsoid out along each free dimension in turn and record the volume.
        log_v = Vector{Float64}(undef, length(free))
        for i in eachindex(free)
            keep = [k for k in eachindex(free) if k != i]
            X = P[keep, :] .- c[keep]
            A_proj = inv(A_inv[keep, keep])
            A_proj ./= maximum(sum(X .* (A_proj * X), dims = 1))
            log_v[i] = -logabsdet(A_proj)[1]          # slogdet(inv(A_proj))
        end

        dim = free[argmin(log_v)]
        dim_cube[dim] = true
        ellipsoid_proj = Ellipsoid(points[.!dim_cube, :]; kw...)

        if log_volume(ellipsoid_proj) < log_volume(ellipsoid)
            ellipsoid = ellipsoid_proj
        else
            dim_cube[dim] = false
            break
        end
    end

    # -- phase two: if that lost to the unit cube, build back up from it ------------------------
    if log_volume(ellipsoid) > 0
        best_log_v = 0.0                              # the unit cube
        ellipsoid = nothing
        dim_cube = trues(n_dim)
        tested = falses(n_dim)

        while !all(tested)
            for dim in findall(!, tested)             # snapshot; `tested` is written below
                dim_cube[dim] = false
                tested[dim] = true
                ellipsoid_test = Ellipsoid(points[.!dim_cube, :]; kw...)
                if best_log_v > log_volume(ellipsoid_test)
                    ellipsoid = ellipsoid_test
                    best_log_v = log_volume(ellipsoid_test)
                    tested[dim_cube] .= false          # re-open the cube dims for another pass
                else
                    dim_cube[dim] = true
                end
            end
        end
    end

    cube = any(dim_cube) ? UnitCube(count(dim_cube); rng) : nothing
    all(dim_cube) && (ellipsoid = nothing)
    return UnitCubeEllipsoidMixture(n_dim, dim_cube, cube, ellipsoid, rng)
end

"""
    transform(b::UnitCubeEllipsoidMixture, points)

Cube dimensions are mapped to `[-1, 1]`, ellipsoid dimensions whitened. This is the frame the
neural bounds and the Gaussian-mixture split both work in.
"""
function transform(b::UnitCubeEllipsoidMixture, points::AbstractMatrix)
    points_t = Matrix{Float64}(points)
    if b.cube !== nothing
        idx = findall(b.dim_cube)
        points_t[idx, :] .= points[idx, :] .* 2 .- 1
    end
    if b.ellipsoid !== nothing
        idx = findall(!, b.dim_cube)
        points_t[idx, :] .= transform(b.ellipsoid, points[idx, :])
    end
    return points_t
end

function in_bound(b::UnitCubeEllipsoidMixture, points::AbstractMatrix)
    res = trues(size(points, 2))
    b.cube !== nothing && (res .&= in_bound(b.cube, points[findall(b.dim_cube), :]))
    b.ellipsoid !== nothing && (res .&= in_bound(b.ellipsoid, points[findall(!, b.dim_cube), :]))
    return res
end

function sample(b::UnitCubeEllipsoidMixture, n_points::Int = 100)
    points = zeros(b.n_dim, n_points)
    b.cube !== nothing && (points[findall(b.dim_cube), :] .= sample(b.cube, n_points))
    b.ellipsoid !== nothing && (points[findall(!, b.dim_cube), :] .= sample(b.ellipsoid, n_points))
    return points
end

log_volume(b::UnitCubeEllipsoidMixture) =
    b.ellipsoid === nothing ? 0.0 : log_volume(b.ellipsoid)

function reset!(b::UnitCubeEllipsoidMixture, rng::AbstractRNG)
    b.rng = rng
    b.cube !== nothing && reset!(b.cube, rng)
    b.ellipsoid !== nothing && reset!(b.ellipsoid, rng)
    return b
end
