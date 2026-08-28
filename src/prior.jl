# Separable priors, specified per parameter.
#
# Port of nautilus/prior.py. A parameter is one of four things: a free parameter with a
# distribution, a free parameter given as a (lo, hi) pair, a value fixed outright, or an alias
# of an earlier parameter. Only the first two consume a dimension of the unit hypercube.
#
# The Python maps the unit hypercube through `dist.isf(1 - u)`, the inverse survival function
# of the complement, rather than the more direct `ppf(u)`; the two are the same function and
# the Julia equivalent of both is `quantile(dist, u)`.

"Marks a parameter aliased to an earlier one."
struct Alias
    key::Symbol
end

"""
    Prior()

An empty prior. Add parameters with [`add_parameter!`](@ref) in the order they should occupy
the unit hypercube.
"""
struct Prior
    keys::Vector{Symbol}
    dists::Vector{Any}
end

Prior() = Prior(Symbol[], Any[])

"""
    add_parameter!(prior, key=nothing, dist=(0, 1))

Add a parameter.

`dist` may be a `(lo, hi)` tuple (uniform), a `Real` (fixed, not sampled), a `Symbol` naming an
earlier parameter (alias), or any `UnivariateDistribution`. A `nothing` key is auto-named
`x_0`, `x_1`, ... by position.
"""
function add_parameter!(p::Prior, key::Union{Nothing,Symbol,AbstractString} = nothing,
                        dist = (0, 1))
    k = if key === nothing
        Symbol("x_$(length(p.keys))")
    else
        s = Symbol(key)
        s in p.keys && throw(ArgumentError("key '$s' is already in the prior"))
        s
    end
    push!(p.keys, k)

    if dist isa Tuple{<:Real,<:Real}
        push!(p.dists, Uniform(dist[1], dist[2]))
    elseif dist isa Real
        push!(p.dists, float(dist))
    elseif dist isa UnivariateDistribution
        push!(p.dists, dist)
    elseif dist isa Union{Symbol,AbstractString}
        target = Symbol(dist)
        (target in p.keys && target != k) ||
            throw(ArgumentError("alias target '$target' was not defined earlier"))
        # Chase alias chains, so every alias points at a real parameter.
        while p.dists[findfirst(==(target), p.keys)] isa Alias
            target = p.dists[findfirst(==(target), p.keys)].key
        end
        push!(p.dists, Alias(target))
    else
        throw(ArgumentError("unsupported 'dist' of type $(typeof(dist))"))
    end
    return p
end

"Number of free parameters, i.e. the dimensionality of the unit hypercube."
dimensionality(p::Prior) = count(d -> d isa UnivariateDistribution, p.dists)

"""
    unit_to_physical(prior, points) -> Matrix

Map `points` from the unit hypercube (one point per column) to physical parameter values.
"""
function unit_to_physical(p::Prior, points::AbstractMatrix{<:Real})
    size(points, 1) == dimensionality(p) ||
        throw(ArgumentError("points have $(size(points, 1)) dimensions, prior has $(dimensionality(p))"))
    out = similar(points, Float64)
    i = 0
    for d in p.dists
        d isa UnivariateDistribution || continue
        i += 1
        @inbounds for j in axes(points, 2)
            out[i, j] = quantile(d, points[i, j])
        end
    end
    return out
end

"""
    physical_to_dictionary(prior, phys_points) -> Dict{Symbol,Vector{Float64}}

Present physical points as one vector per named parameter, fixed values and aliases included.
"""
function physical_to_dictionary(p::Prior, phys_points::AbstractMatrix{<:Real})
    size(phys_points, 1) == dimensionality(p) ||
        throw(ArgumentError("points have $(size(phys_points, 1)) dimensions, prior has $(dimensionality(p))"))
    n = size(phys_points, 2)
    out = Dict{Symbol,Vector{Float64}}()

    i = 0
    for (key, d) in zip(p.keys, p.dists)
        if d isa UnivariateDistribution
            i += 1
            out[key] = phys_points[i, :]
        elseif d isa Float64
            out[key] = fill(d, n)
        end
    end
    # Aliases resolve after the pass above, since they may point anywhere earlier.
    for (key, d) in zip(p.keys, p.dists)
        d isa Alias && (out[key] = out[d.key])
    end
    return out
end

unit_to_dictionary(p::Prior, points::AbstractMatrix{<:Real}) =
    physical_to_dictionary(p, unit_to_physical(p, points))
