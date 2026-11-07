# Periodic parameters, by phase shift.
#
# Port of nautilus/bounds/periodic.py.
#
# For each periodic dimension, the points above the threshold are rotated around the circle
# so that the widest gap between them lands on the 0/1 seam. Every geometric bound is then
# built in the rotated frame, where a mode that straddled the seam is in one piece, and points
# are rotated back on the way out. The rotation preserves volume, so nothing else in the
# method has to change.
#
# The method is deliberately simple: no periodic features in the network, no wrap-aware
# distances. It is enough for well-separated modes on the circle, since the widest gap then
# always lies between two modes and the seam never cuts one. Its limit is a live set that
# fills nearly the whole circle: the seam then has to cut through points, and an ellipsoid
# enlarged past the cube edge is clipped there. A bound at that stage is wide anyway, so the
# clipping costs efficiency rather than correctness.

struct PhaseShift
    periodic::Vector{Int}
    centers::Vector{Float64}
end

"""
    PhaseShift(points, periodic)

For each dimension in `periodic`, find the widest gap between `points` (one per column) on
the unit circle, and choose the rotation that moves the middle of that gap to the 0/1 seam.
Port of `PhaseShift.compute` (periodic.py:22).
"""
function PhaseShift(points::AbstractMatrix{<:Real}, periodic::AbstractVector{<:Integer})
    centers = zeros(length(periodic))
    for (i, dim) in enumerate(periodic)
        x = sort(points[dim, :])
        # The gaps between neighbours, plus the one that wraps from the last point to the first.
        dx = push!(diff(x), x[1] - (x[end] - 1))
        k = argmax(dx)
        centers[i] = mod(x[k] + dx[k] / 2 + 0.5, 1)
    end
    return PhaseShift(collect(Int, periodic), centers)
end

# `mod(-1e-20, 1.0)` rounds to exactly 1.0. On the circle 1 is 0, and 0 is the value that
# stays inside a half-open prior range such as [0, 360).
_wrap(x::Float64) = (y = mod(x, 1.0); y == 1.0 ? 0.0 : y)

"""
    transform(shift::PhaseShift, points; inverse=false)

Apply the phase shift to a copy of `points`, or undo it with `inverse = true`.
"""
function transform(shift::PhaseShift, points::AbstractMatrix; inverse::Bool = false)
    out = Matrix{Float64}(points)
    sgn = inverse ? -1.0 : 1.0
    for (i, dim) in enumerate(shift.periodic)
        delta = sgn * (0.5 - shift.centers[i])
        @inbounds for j in axes(out, 2)
            out[dim, j] = _wrap(out[dim, j] + delta)
        end
    end
    return out
end
