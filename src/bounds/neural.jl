# The neural bound: an ellipsoid, refined by a network that predicts whether a point clears the
# likelihood threshold.
#
# Port of nautilus/bounds/neural.py.

"""
    polyval_at(x, y, deg, x0)

Fit a degree-`deg` polynomial to `(x, y)` by least squares and evaluate it at `x0`.

This is the one place nautilus uses `np.polyfit`/`np.polyval` (neural.py:96), for a cubic.
A Vandermonde solve is the whole of it, so it needs no polynomial package.
"""
function polyval_at(x::AbstractVector{Float64}, y::AbstractVector{Float64}, deg::Int, x0::Real)
    V = hcat((x .^ d for d in 0:deg)...)
    c = V \ y
    return sum(c[d + 1] * x0^d for d in 0:deg)
end

"""
    NeuralBound

An outer `Ellipsoid` around the points above the likelihood threshold, intersected with the
region where the emulator predicts a score above `score_predict_min`.
"""
mutable struct NeuralBound
    n_dim::Int
    outer_bound::Ellipsoid
    emulator::Union{NeuralNetworkEmulator,Nothing}
    score_predict_min::Float64
end

"""
    NeuralBound(points, log_l, log_l_min; enlarge_per_dim=1.1, n_networks=4, rng, kwargs...)

Port of `NeuralBound.compute` (neural.py:29).

The network is a *classifier in disguise*. Its target is a rank score in `[0, 1]`: points above
the threshold are ranked among themselves and mapped into `(0.5, 1)`, points below into
`(0, 0.5)`. Training uses every point inside the outer ellipsoid, the below-threshold ones
included — they are what teach the network where the boundary is.

The acceptance threshold is then read off a cubic fit of predicted score against true score,
evaluated at the true score of the lowest above-threshold point. Fitting predicted-against-true
rather than thresholding at 0.5 is what absorbs the network's systematic bias.

`n_networks = 0` skips the network entirely and leaves a plain ellipsoid.
"""
function NeuralBound(points::AbstractMatrix{<:Real}, log_l::AbstractVector{<:Real},
                     log_l_min::Real; enlarge_per_dim::Real = 1.1, n_networks::Int = 4,
                     rng::AbstractRNG = Random.default_rng(), kwargs...)
    n_dim = size(points, 1)
    pts = Matrix{Float64}(points)
    ll = Vector{Float64}(log_l)

    outer = Ellipsoid(pts[:, ll .>= log_l_min]; enlarge_per_dim, rng)
    n_networks == 0 && return NeuralBound(n_dim, outer, nothing, 0.0)

    sel = in_bound(outer, pts)
    pts = pts[:, sel]
    ll = ll[sel]
    pts_t = transform(outer, pts)

    above = ll .>= log_l_min
    n_above, n_below = count(above), count(!, above)
    score = zeros(length(ll))
    score[above] .= 0.5 .* (1 .+ (tiedrank(ll[above]) .- 0.5) ./ n_above)
    score[.!above] .= 0.5 .* ((tiedrank(ll[.!above]) .- 0.5) ./ n_below)

    emulator = train_emulator(pts_t, score; n_networks, kwargs...)
    score_predict_min = polyval_at(score, predict(emulator, pts_t), 3, minimum(score[above]))

    return NeuralBound(n_dim, outer, emulator, score_predict_min)
end

function in_bound(b::NeuralBound, points::AbstractMatrix)
    res = in_bound(b.outer_bound, points)
    if any(res) && b.emulator !== nothing
        pred = predict(b.emulator, transform(b.outer_bound, points[:, res]))
        # The 1e-9 slack keeps a network that has collapsed to a near-constant output from
        # rejecting everything (neural.py:124).
        res[res] .= pred .> b.score_predict_min - 1e-9
    end
    return res
end
