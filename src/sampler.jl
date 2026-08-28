# The sampler: shells, importance weights, and the main exploration/sampling loop.
#
# Port of nautilus/sampler.py.
#
# The method builds a nested sequence of bounds B_1 ⊃ B_2 ⊃ ... , each approximating a region
# of increasing likelihood. Shell i is the part of B_i not covered by any later bound, so the
# shells partition the space. Each shell carries its own volume estimate and its own points,
# and the evidence is the volume-weighted sum of likelihoods over shells. Because points are
# drawn from a known bound rather than evolved, every point ever evaluated stays in the
# estimate -- that is the "importance" in importance nested sampling, and it is why the
# posterior sample is so much larger than a classical nested sampler's.
#
# Not ported (deliberately, see the plan): HDF5 checkpointing and resume, blobs, and periodic
# parameters.

const _STATUS_COLS = ("Status", "Bounds", "Ellipses", "Networks", "Calls", "f_live", "N_eff", "log Z")
const _STATUS_WIDTHS = (9, 6, 8, 8, 8, 6, 5, 7)

mutable struct Sampler{P,L}
    prior::P
    likelihood::L
    n_dim::Int
    n_live::Int
    n_update::Int
    enlarge_per_dim::Float64
    n_points_min::Int
    split_threshold::Float64
    n_networks::Int
    n_batch::Int
    n_like_new_bound::Int
    vectorized::Bool
    pass_dict::Bool
    threaded::Bool
    rng::AbstractRNG

    n_like::Int
    explored::Bool
    discard_exploration::Bool

    bounds::Vector{Union{UnitCube,NautilusBound}}
    points::Vector{Matrix{Float64}}
    log_l::Vector{Vector{Float64}}

    shell_n::Vector{Int}
    shell_n_sample::Vector{Int}
    shell_n_eff::Vector{Float64}
    shell_log_l_min::Vector{Float64}
    shell_log_l::Vector{Float64}
    shell_log_v::Vector{Float64}
    shell_n_sample_exp::Vector{Int}
    shell_end_exp::Vector{Int}

    points_t::Matrix{Float64}
    shell_t::Vector{Int}
    log_l_t::Vector{Float64}

    n_update_iter::Int
    n_like_iter::Int
end

"""
    Sampler(prior, likelihood; kwargs...)

Build a sampler. `prior` is either a [`Prior`](@ref) or a callable mapping a point in the unit
hypercube to physical parameters; `likelihood` returns a log-likelihood.

Keywords follow the Python defaults (sampler.py:121). Note `n_live = 2000` — the Python
docstring says 3000, but the code says 2000 and the code is what runs.

  * `n_live = 2000` — points kept in the live set; the main accuracy knob.
  * `n_update = n_live` — points above the threshold before a new bound is built.
  * `n_like_new_bound = 10 * n_live` — likelihood calls before a new bound is forced.
  * `n_points_min = n_dim + 50` — minimum points per ellipsoid.
  * `enlarge_per_dim = 1.1`, `split_threshold = 100`, `n_networks = 4`.
  * `vectorized = false` — if true, `likelihood` receives all points at once.
  * `threaded = false` — evaluate the likelihood, and draw from the bounds, on all available
    threads. **Opt-in, because it requires a thread-safe `likelihood`**: one that closes over
    a shared preallocated workspace (a common pattern in model-fitting code) will silently
    corrupt its results. Reproducibility then holds per seed *and* thread count.
  * `pass_dict` — pass parameters as a `Dict`; defaults to true for a `Prior`, false otherwise.
"""
function Sampler(prior, likelihood; n_dim::Union{Nothing,Integer} = nothing,
                 n_live::Integer = 2000, n_update::Union{Nothing,Integer} = nothing,
                 enlarge_per_dim::Real = 1.1, n_points_min::Union{Nothing,Integer} = nothing,
                 split_threshold::Real = 100, n_networks::Integer = 4,
                 n_batch::Union{Nothing,Integer} = nothing,
                 n_like_new_bound::Union{Nothing,Integer} = nothing,
                 vectorized::Bool = false, pass_dict::Union{Nothing,Bool} = nothing,
                 threaded::Bool = false, seed::Union{Nothing,Integer} = nothing)

    d = if prior isa Prior
        dimensionality(prior)
    else
        n_dim === nothing &&
            throw(ArgumentError("'n_dim' is required when 'prior' is a callable"))
        Int(n_dim)
    end
    d <= 1 && throw(ArgumentError("cannot run Nautilus with fewer than 2 parameters, got $d"))

    pd = pass_dict === nothing ? (prior isa Prior) : pass_dict
    pd && !(prior isa Prior) &&
        throw(ArgumentError("'pass_dict' requires a Prior, not a callable"))

    nu = n_update === nothing ? Int(n_live) : Int(n_update)
    nlnb = n_like_new_bound === nothing ? 10 * Int(n_live) : Int(n_like_new_bound)
    npm = n_points_min === nothing ? d + 50 : Int(n_points_min)
    nb = n_batch === nothing ? 100 : Int(n_batch)
    rng = seed === nothing ? Random.default_rng() : Xoshiro(seed)

    return Sampler{typeof(prior),typeof(likelihood)}(
        prior, likelihood, d, Int(n_live), nu, Float64(enlarge_per_dim), npm,
        Float64(split_threshold), Int(n_networks), nb, nlnb, vectorized, pd, threaded, rng,
        0, false, false,
        Union{UnitCube,NautilusBound}[], Matrix{Float64}[], Vector{Float64}[],
        Int[], Int[], Float64[], Float64[], Float64[], Float64[], Int[], Int[],
        zeros(d, 0), Int[], Float64[], 0, 0)
end

# --------------------------------------------------------------------------------------------
# Likelihood evaluation
# --------------------------------------------------------------------------------------------

"""
    evaluate_likelihood(s, points) -> Vector{Float64}

Map `points` (unit hypercube, one per column) through the prior and evaluate the likelihood.
Increments the call counter.
"""
# Apply `f` to each index, on all threads when asked. One helper so the serial and threaded
# paths cannot drift apart.
function _map_indices(f, n::Int, threaded::Bool)
    if threaded && Threads.nthreads() > 1
        out = Vector{Float64}(undef, n)
        Threads.@threads for i in 1:n
            out[i] = f(i)
        end
        return out
    end
    return Float64[f(i) for i in 1:n]
end

function evaluate_likelihood(s::Sampler, points::AbstractMatrix{Float64})
    n = size(points, 2)
    n == 0 && return Float64[]

    log_l = if s.prior isa Prior && s.pass_dict
        dict = unit_to_dictionary(s.prior, points)
        s.vectorized ? s.likelihood(dict) :
            _map_indices(i -> s.likelihood(Dict(k => v[i] for (k, v) in dict)), n, s.threaded)
    else
        phys = if s.prior isa Prior
            unit_to_physical(s.prior, points)
        elseif s.vectorized
            s.prior(points)
        else
            reduce(hcat, (s.prior(points[:, i]) for i in 1:n))
        end
        s.vectorized ? s.likelihood(phys) :
            _map_indices(i -> s.likelihood(phys[:, i]), n, s.threaded)
    end

    s.n_like += n
    return Vector{Float64}(log_l)
end

# --------------------------------------------------------------------------------------------
# Shell bookkeeping
# --------------------------------------------------------------------------------------------

"""
    update_shell_info!(s, index)

Recompute a shell's volume, mean likelihood and effective sample size. Port of
`update_shell_info` (sampler.py:921).

`shell_n_eff` is Kish's effective sample size computed in log space, which is what keeps it
stable when the likelihoods span hundreds of e-folds.
"""
function update_shell_info!(s::Sampler, index::Int)
    n_sample = s.shell_n_sample[index]
    start = 0
    if s.discard_exploration && s.explored
        start = s.shell_end_exp[index]
        n_sample -= s.shell_n_sample_exp[index]
    end

    log_l = s.log_l[index][(start + 1):end]
    n = length(log_l)
    s.shell_n[index] = n

    if n > 0
        s.shell_log_v[index] = log_volume(s.bounds[index]) + log(n / n_sample)
        s.shell_log_l[index] = logsumexp(log_l) - log(n)
        s.shell_n_eff[index] = if all(==(-Inf), log_l)
            float(n)
        else
            exp(2 * logsumexp(log_l) - logsumexp(2 .* log_l))
        end
    else
        s.shell_log_v[index] = -Inf
        s.shell_log_l[index] = NaN
        s.shell_n_eff[index] = 0.0
    end
    return nothing
end

"""
    shell_association(s, points; n_max=length(s.bounds)) -> Vector{Int}

The highest-index bound (up to `n_max`) containing each point, or `0` for none.
"""
function shell_association(s::Sampler, points::AbstractMatrix; n_max::Int = length(s.bounds))
    shell = zeros(Int, size(points, 2))
    for i in n_max:-1:1
        un = findall(iszero, shell)
        isempty(un) && break
        inb = in_bound(s.bounds[i], points[:, un])
        shell[un[inb]] .= i
    end
    return shell
end

"""
    shell_bound_occupation(s; fractional=true) -> Matrix

`m[i, k]` is the number of shell-`i` points also inside bound `k`. A well-behaved run gives a
lower-triangular matrix of ones: every bound contains all the later shells. See
`docs/discussion/convergence.rst` in the Python package.
"""
function shell_bound_occupation(s::Sampler; fractional::Bool = true)
    n = length(s.bounds)
    m = zeros(Float64, n, n)
    for i in 1:n, k in 1:n
        m[i, k] = count(in_bound(s.bounds[k], s.points[i]))
    end
    fractional && (m ./= diag(m))
    return m
end

# Per-point log volumes, repeated out shell by shell.
function _log_v_per_point(s::Sampler)
    # `reduce(vcat, ...)` rather than a splat: splatting an empty collection infers as
    # Vector{Any}, which JET flags and which poisons every caller downstream.
    return reduce(vcat, (fill(s.shell_log_v[i] - log(max(s.shell_n[i], 1)), s.shell_n[i])
                         for i in eachindex(s.shell_n)); init = Float64[])
end

"""
    log_z(s) -> Float64

Log of the Bayesian evidence: the volume-weighted sum of shell likelihoods.
"""
function log_z(s::Sampler)
    sum(s.shell_n) == 0 && return NaN
    sel = .!isnan.(s.shell_log_l)
    return logsumexp(s.shell_log_l[sel] .+ s.shell_log_v[sel])
end

"""
    n_eff(s) -> Float64

Effective posterior sample size, combined across shells (sampler.py:661).
"""
function n_eff(s::Sampler)
    all(iszero, s.shell_n_eff) && return 0.0
    sel = s.shell_n_eff .> 0
    z = s.shell_log_l .+ s.shell_log_v
    w = exp.(z .- maximum(filter(!isnan, z)))[sel]
    return sum(w)^2 / sum(w .^ 2 ./ s.shell_n_eff[sel])
end

"""
    eta(s) -> Float64

Asymptotic sampling efficiency, the limit of `N_eff / N_like` (sampler.py:721).
"""
function eta(s::Sampler)
    sel = .!isnan.(s.shell_log_l)
    log_z_s = (s.shell_log_l .+ s.shell_log_v)[sel]
    shell_eta = (s.shell_n_eff ./ s.shell_n)[sel]
    return exp(2 * logsumexp(log_z_s) - 2 * logsumexp(log_z_s .- 0.5 .* log.(shell_eta)))
end

"""
    f_live(s) -> Float64

Fraction of the evidence still sitting in the live set; the exploration phase stops when this
drops below the target.
"""
function f_live(s::Sampler)
    s.explored && return NaN
    sum(s.shell_n) == 0 && return 1.0
    log_l = reduce(vcat, s.log_l; init = Float64[])
    log_w = _log_v_per_point(s) .+ log_l
    live = log_w[sortperm(log_l)][max(end - s.n_live + 1, 1):end]
    return exp(logsumexp(live) - logsumexp(log_w))
end

"Volume currently held by the live set — the target handed to each new bound."
function log_v_live(s::Sampler)
    isempty(s.bounds) && return 1.0
    log_l = reduce(vcat, s.log_l; init = Float64[])
    log_v = _log_v_per_point(s)
    return logsumexp(log_v[sortperm(log_l)][max(end - s.n_live + 1, 1):end])
end

# --------------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------------

"""
    sample_shell(s, index; transfer=false)

Draw `n_batch` points that fall in shell `index` — inside bound `index`, outside every later
bound. Port of `sample_shell` (sampler.py:762).

When `transfer` is set (only valid for the newest shell), freshly drawn points are matched
against already-evaluated points held in the transfer buffer and dropped in their favour. That
is where much of nautilus's efficiency comes from: those points cost no likelihood call. The
draw is still counted in `n_bound`, so the volume estimate is unaffected.
"""
function sample_shell(s::Sampler, index::Int; transfer::Bool = false)
    n_bound = 0
    n_sample = 0
    idx_t = Int[]
    collected = Matrix{Float64}[]

    while n_sample < s.n_batch
        want = s.n_batch - n_sample
        pts = s.bounds[index] isa NautilusBound ?
              sample(s.bounds[index], want; threaded = s.threaded) :
              sample(s.bounds[index], want)
        n_bound += want

        in_shell = trues(size(pts, 2))
        for b in s.bounds[(index + 1):end]
            in_shell .&= .!in_bound(b, pts)
        end
        pts = pts[:, in_shell]

        replace = falses(size(pts, 2))
        if transfer && !isempty(s.shell_t)
            shell_p = shell_association(s, pts; n_max = length(s.bounds) - 1)
            for shell in 1:(length(s.bounds) - 1)
                idx_1 = findall(==(shell), s.shell_t)
                idx_2 = findall(==(shell), shell_p)
                n = min(length(idx_1), length(idx_2))
                n == 0 && continue
                chosen = idx_1[randperm(s.rng, length(idx_1))[1:n]]
                append!(idx_t, chosen)
                s.shell_t[chosen] .= -1        # spent, so it cannot be matched twice
                replace[idx_2[randperm(s.rng, length(idx_2))[1:n]]] .= true
            end
        end

        pts = pts[:, .!replace]
        if size(pts, 2) > 0
            push!(collected, pts)
            n_sample += size(pts, 2)
        end
    end

    points = reduce(hcat, collected; init = zeros(s.n_dim, 0))
    return points, n_bound, idx_t
end

"""
    add_bound!(s; verbose=false) -> Bool

Build the next bound, at the likelihood threshold of the `n_live`-th best point. Port of
`add_bound` (sampler.py:993).

Returns false, leaving the run to continue with the current bounds, when the new bound would
not actually be smaller than the last one, or when there is nothing below the threshold to
zoom in on.
"""
function add_bound!(s::Sampler; verbose::Bool = false)
    local log_l_min::Float64

    if isempty(s.bounds)
        log_l_min = -Inf
        push!(s.bounds, UnitCube(s.n_dim; rng = s.rng))
        success = true
    else
        verbose && print_status(s, "Bounding"; newline = false)
        all_log_l = reduce(vcat, s.log_l; init = Float64[])
        perm = sortperm(all_log_l)
        pts = reduce(hcat, s.points)[:, perm]
        ll = all_log_l[perm]
        log_l_min = ll[end - s.n_live + 1]

        # A threshold landing on a likelihood plateau would make the bound meaningless, so
        # step up to the next distinct value when there is still enough room above it.
        if count(==(log_l_min), ll) > 1 && count(>(log_l_min), ll) >= s.n_points_min
            log_l_min = minimum(ll[ll .> log_l_min])
        end

        if all(ll .>= log_l_min)
            success = false
        else
            bound = NautilusBound(pts, ll, log_l_min, log_v_live(s);
                                  enlarge_per_dim = s.enlarge_per_dim,
                                  n_points_min = s.n_points_min,
                                  split_threshold = s.split_threshold,
                                  n_networks = s.n_networks, rng = s.rng)
            sample(bound, 1000; return_points = false, threaded = s.threaded)
            if log_volume(bound) < log_volume(s.bounds[end])
                push!(s.bounds, bound)
                success = true
            else
                success = false
            end
        end
    end

    if !success
        s.shell_log_l_min[end] = log_l_min
        return false
    end

    push!(s.shell_n, 0)
    push!(s.shell_n_sample, 0)
    push!(s.shell_n_eff, 0.0)
    push!(s.shell_log_l, NaN)
    push!(s.shell_log_v, NaN)
    push!(s.shell_log_l_min, log_l_min)
    push!(s.points, zeros(s.n_dim, 0))
    push!(s.log_l, Float64[])

    # Points already evaluated that fall inside the new bound move to the transfer buffer,
    # ready to be swapped in for points the next batch would otherwise have to evaluate.
    if length(s.bounds) > 1
        shell_t = Int[]
        points_t = Matrix{Float64}[]
        log_l_t = Float64[]
        for shell in 1:(length(s.bounds) - 1)
            inb = in_bound(s.bounds[end], s.points[shell])
            append!(shell_t, fill(shell, count(inb)))
            push!(points_t, s.points[shell][:, inb])
            append!(log_l_t, s.log_l[shell][inb])
            s.points[shell] = s.points[shell][:, .!inb]
            s.log_l[shell] = s.log_l[shell][.!inb]
            s.shell_n[shell] -= count(inb)
            update_shell_info!(s, shell)
        end
        s.shell_t = shell_t
        s.points_t = reduce(hcat, points_t; init = zeros(s.n_dim, 0))
        s.log_l_t = log_l_t
    end

    return true
end

"""
    add_samples!(s, shell; verbose=false) -> Int

Add one batch to `shell` (use `shell = 0` for the newest one, where transfers apply). Returns
how many of the new points cleared that shell's likelihood threshold.
"""
function add_samples!(s::Sampler, shell::Int; verbose::Bool = false)
    verbose && print_status(s, "Sampling"; newline = false)
    newest = shell == 0
    idx = newest ? length(s.bounds) : shell

    points, n_bound, idx_t = sample_shell(s, idx; transfer = newest && !isempty(s.shell_t))
    verbose && print_status(s, "Computing"; newline = false)

    if !isempty(idx_t)
        s.points[idx] = hcat(s.points[idx], s.points_t[:, idx_t])
        append!(s.log_l[idx], s.log_l_t[idx_t])
    end

    s.shell_n_sample[idx] += n_bound
    log_l = evaluate_likelihood(s, points)
    s.points[idx] = hcat(s.points[idx], points)
    append!(s.log_l[idx], log_l)
    update_shell_info!(s, idx)

    return count(>=(s.shell_log_l_min[idx]), log_l)
end

# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------

function print_status(s::Sampler, status::String = ""; header::Bool = false,
                      newline::Bool = true)
    if header
        println(join((rpad(c, w) for (c, w) in zip(_STATUS_COLS, _STATUS_WIDTHS)), "| "))
        return nothing
    end
    nb = length(s.bounds)
    ne = (nb > 0 && s.bounds[end] isa NautilusBound) ? n_ell(s.bounds[end]) : 0
    nn = (nb > 0 && s.bounds[end] isa NautilusBound) ? n_net(s.bounds[end]) : 0
    fl = s.explored ? NaN : f_live(s)
    lz = log_z(s)
    fields = (status, string(nb), string(ne), string(nn), string(s.n_like),
              isnan(fl) ? "N/A" : @sprintf("%.4f", fl),
              @sprintf("%.0f", n_eff(s)),
              isnan(lz) ? "N/A" : @sprintf("%+.2f", lz))
    print(join((rpad(f, w) for (f, w) in zip(fields, _STATUS_WIDTHS)), "| "))
    print(newline ? "\n" : "\r")
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Live parameter display
# ---------------------------------------------------------------------------------------------
#
# Modelled on UltraNest's `ultranest.viz.nicelogger`, which nautilus has no equivalent of --
# upstream's entire console output is the one status line above. The idea is worth having: a
# bar per parameter spanning the prior range, with the bins the live set currently occupies
# marked, so the intervals are visibly narrowing as the run proceeds.
#
# Where UltraNest labels bins by MultiNest-style cluster, this labels them by which ellipsoid
# of the latest bound's outer union the point falls in, which is the same information in
# nautilus's terms: a run that has resolved two modes draws `1` and `2` in separate places.

const _MODE_CHARS = collect("123456789abcdefghijklmnopqrstuvwxyz")

_fmtnum(v) = (v != 0 && (abs(v) >= 1e4 || abs(v) < 1e-3)) ?
             @sprintf("%+.1e", v) : @sprintf("%+.3f", v)

function _param_names(s::Sampler)
    s.prior isa Prior || return ["x_$i" for i in 1:s.n_dim]
    return [string(k) for (k, d) in zip(s.prior.keys, s.prior.dists)
            if d isa UnivariateDistribution]
end

# Physical coordinates of the current live set (the n_live highest-likelihood points), plus
# the same points in unit-cube space, which is what the bounds test against.
function _live_points(s::Sampler)
    log_l = reduce(vcat, s.log_l; init = Float64[])
    isempty(log_l) && return zeros(s.n_dim, 0), zeros(s.n_dim, 0)
    pts = reduce(hcat, s.points; init = zeros(s.n_dim, 0))
    k = min(s.n_live, length(log_l))
    unit = pts[:, partialsortperm(log_l, 1:k, rev = true)]
    phys = if s.prior isa Prior
        unit_to_physical(s.prior, unit)
    else
        reduce(hcat, (s.prior(unit[:, i]) for i in axes(unit, 2)); init = zeros(s.n_dim, 0))
    end
    return phys, unit
end

# The prior range to draw against. Taken at u = 0.001 and 0.999 rather than 0 and 1 so that an
# unbounded prior (a Normal, say) gives a finite bar instead of an infinite one.
function _display_range(s::Sampler)
    lo_u, hi_u = fill(0.001, s.n_dim), fill(0.999, s.n_dim)
    if s.prior isa Prior
        return vec(unit_to_physical(s.prior, reshape(lo_u, :, 1))),
               vec(unit_to_physical(s.prior, reshape(hi_u, :, 1)))
    end
    return Vector{Float64}(s.prior(lo_u)), Vector{Float64}(s.prior(hi_u))
end

# Which mode each live point belongs to; 0 if none, all zeros when the only bound is the unit
# cube.
#
# Keyed off the INNER union -- one ellipsoid per neural bound -- not the outer one. The inner
# union is split until no two ellipsoids overlap, so it is what actually resolves modes; the
# outer union splits only when its volume exceeds the target by `split_threshold`, and on a
# clean bimodal problem it can happily stay a single ellipsoid spanning both peaks.
function _mode_ids(s::Sampler, unit::AbstractMatrix)
    ids = zeros(Int, size(unit, 2))
    (isempty(s.bounds) || !(s.bounds[end] isa NautilusBound)) && return ids
    nb = s.bounds[end].neural_bounds
    ells = isempty(nb) ? s.bounds[end].outer_bound.bounds : [b.outer_bound for b in nb]
    for (k, sub) in enumerate(ells)
        inb = in_bound(sub, unit)
        @inbounds for j in eachindex(ids)
            ids[j] == 0 && inb[j] && (ids[j] = k)
        end
    end
    return ids
end

"""
    print_live_display(s; io=stdout, width=42) -> Int

Draw one bar per parameter showing where the live set sits inside the prior range. Returns the
number of lines written, so a caller redrawing in place knows how far to move the cursor back.

Occupied bins are marked `*`, or `1`, `2`, ... when the newest bound has resolved more than one
ellipsoid. The flanking numbers are the prior range; the numbers inside the bar, where there is
room for them, are the live set's current extent — the pair that visibly closes in as the run
proceeds.
"""
function print_live_display(s::Sampler; io::IO = stdout, width::Int = 42)
    phys, unit = _live_points(s)
    size(phys, 2) == 0 && return 0

    lo, hi = _display_range(s)
    names = _param_names(s)
    ids = _mode_ids(s, unit)
    nmodes = maximum(ids; init = 0)
    pad = maximum(length, names; init = 1)
    lines = 0

    for i in 1:s.n_dim
        l, h = lo[i], hi[i]
        h > l || continue
        x = @view phys[i, :]
        bin(v) = clamp(1 + floor(Int, (v - l) / (h - l) * width), 1, width)

        bar = fill(' ', width)
        @inbounds for j in eachindex(x)
            b = bin(x[j])
            c = (nmodes > 1 && ids[j] > 0) ?
                _MODE_CHARS[min(ids[j], length(_MODE_CHARS))] : '*'
            bar[b] == ' ' && (bar[b] = c)
        end

        # Inline the live extent where the empty margin is wide enough to hold it, exactly as
        # UltraNest does -- it is what makes the narrowing legible without reading the axis.
        xlo, xhi = extrema(x)
        ilo, ihi = bin(xlo), bin(xhi)
        ls, rs = _fmtnum(xlo), _fmtnum(xhi)
        if ilo - 2 - length(ls) >= 1
            bar[(ilo - 1 - length(ls)):(ilo - 2)] .= collect(ls)
        end
        if ihi + 1 + length(rs) <= width
            bar[(ihi + 2):(ihi + 1 + length(rs))] .= collect(rs)
        end

        println(io, "  ", rpad(names[i], pad), " ", lpad(_fmtnum(l), 9),
                " |", String(bar), "| ", _fmtnum(h))
        lines += 1
    end
    return lines
end

"""
    set_discard_exploration!(s, value)

Discard the points collected during the exploration phase. Those points come from a
pseudo-importance function and carry a small bias; dropping them removes it, at the cost of a
smaller sample. The Python docs recommend it for published results.
"""
function set_discard_exploration!(s::Sampler, value::Bool)
    s.discard_exploration = value
    for i in eachindex(s.bounds)
        update_shell_info!(s, i)
    end
    return s
end

"""
    run!(s; f_live=0.01, n_shell=1, n_eff=10000, n_like_max=Inf,
         discard_exploration=false, timeout=Inf, verbose=false) -> Bool

Run the sampler to convergence. Returns whether it converged, as opposed to stopping on
`n_like_max` or `timeout`.

There are two phases. **Exploration** repeatedly adds a bound at a higher likelihood threshold
and fills it, until the live set holds less than `f_live` of the evidence. **Sampling** then
tops up whichever shell most improves the result, chosen by
`shell_log_l + shell_log_v - log(shell_n)/2 - log(shell_n_eff)/2`, until `n_eff` is reached.

`live_display = true` (with `verbose`) replaces the single status line with a redrawn block
showing, per parameter, where the live set sits inside the prior range — see
[`print_live_display`](@ref). It needs a terminal and falls back to the plain status line when
output is redirected.
"""
function run!(s::Sampler; f_live::Real = 0.01, n_shell::Integer = 1, n_eff::Real = 10_000,
              n_like_max::Real = Inf, discard_exploration::Bool = false,
              timeout::Real = Inf, verbose::Bool = false, live_display::Bool = false)
    t_start = time()
    # `n_eff` and `f_live` are both keyword names here and function names in this module, so
    # the targets are captured before the keywords shadow them.
    n_eff_target = n_eff
    f_live_target = f_live

    # The live display redraws a block in place, which needs a terminal; in a log or a pipe it
    # would emit a screenful of escape codes per batch, so it degrades to the plain status
    # line there rather than being silently broken.
    show_live = live_display && verbose && (stdout isa Base.TTY)
    inner_verbose = verbose && !show_live      # the block owns the cursor when it is drawn
    drawn = Ref(0)

    function redraw(status::AbstractString)
        show_live || return
        drawn[] > 0 && print("\e[", drawn[], "A")
        print_status(s, status)
        drawn[] = 1 + print_live_display(s)
    end

    if verbose
        println(s.n_like == 0 ? "Starting the nautilus sampler..." : "Resuming nautilus run...")
        print_status(s; header = true)
    end

    if isempty(s.bounds)
        add_bound!(s)
        s.n_update_iter = -s.n_live
        s.n_like_iter = 0
    end

    converged() = s.explored && all(>=(n_shell), s.shell_n) && Nautilus.n_eff(s) >= n_eff_target
    success = converged()

    while s.n_like < n_like_max && (time() - t_start) < timeout && !success
        if !s.explored
            if (s.n_update_iter >= s.n_update || s.n_like_iter >= s.n_like_new_bound) &&
               sum(s.shell_n) > s.n_live
                add_bound!(s; verbose = inner_verbose)
                s.n_update_iter = 0
                s.n_like_iter = 0
            end

            s.n_update_iter += add_samples!(s, 0; verbose = inner_verbose)
            s.n_like_iter += s.n_batch

            if Nautilus.f_live(s) <= f_live_target
                # Empty shells hold ~no volume and may never yield a point; drop them.
                for shell in reverse(findall(iszero, s.shell_n))
                    deleteat!(s.bounds, shell)
                    deleteat!(s.points, shell)
                    deleteat!(s.log_l, shell)
                    for v in (s.shell_n, s.shell_n_sample, s.shell_n_eff,
                              s.shell_log_l_min, s.shell_log_l, s.shell_log_v)
                        deleteat!(v, shell)
                    end
                end
                s.shell_n_sample_exp = copy(s.shell_n_sample)
                s.shell_end_exp = [size(p, 2) for p in s.points]
                s.explored = true
                set_discard_exploration!(s, discard_exploration)
            end

        elseif any(<(n_shell), s.shell_n)
            add_samples!(s, findfirst(<(n_shell), s.shell_n); verbose = inner_verbose)
        else
            score = s.shell_log_l .+ s.shell_log_v .- 0.5 .* log.(s.shell_n) .-
                    0.5 .* log.(s.shell_n_eff)
            add_samples!(s, argmax(replace(score, NaN => -Inf)); verbose = inner_verbose)
        end

        success = converged()
        redraw(s.explored ? "Sampling" : "Exploring")
    end

    if show_live
        redraw(success ? "Finished" : "Stopped")
    elseif verbose
        print_status(s, success ? "Finished" : "Stopped")
    end
    return success
end

"""
    posterior(s; equal_weight=false, equal_weight_boost=1.0, return_as_dict=false)

Posterior samples as `(points, log_w, log_l)`, with `points` in **(n_samples, n_params)** — the
transpose of the internal layout, matching what corner-plot packages expect. `log_w` is
normalised to sum to one in linear space.

With `equal_weight`, points are resampled by stochastic rounding of their weights so that all
weights are equal, which is the form PairPlots.jl and most corner-plot code want.
`equal_weight_boost` scales the multiplicities up for a larger, less lossy sample.
"""
function posterior(s::Sampler; equal_weight::Bool = false, equal_weight_boost::Real = 1.0,
                   return_as_dict::Bool = false)
    start = (s.discard_exploration && s.explored) ? s.shell_end_exp : zeros(Int, length(s.points))

    points = reduce(hcat, (p[:, (st + 1):end] for (p, st) in zip(s.points, start));
                    init = zeros(s.n_dim, 0))
    log_l = reduce(vcat, (ll[(st + 1):end] for (ll, st) in zip(s.log_l, start)); init = Float64[])
    log_w = _log_v_per_point(s) .+ log_l

    if equal_weight
        r = exp.(log_w .- maximum(log_w)) .* equal_weight_boost
        reps = floor.(Int, r) .+ (rand(s.rng, length(r)) .< (r .- floor.(r)))
        points = reduce(hcat, (repeat(points[:, i:i], 1, reps[i]) for i in eachindex(reps)
                               if reps[i] > 0); init = zeros(s.n_dim, 0))
        log_l = reduce(vcat, (fill(log_l[i], reps[i]) for i in eachindex(reps)); init = Float64[])
        log_w = zeros(sum(reps))
    end

    phys = if s.prior isa Prior
        return_as_dict ? unit_to_dictionary(s.prior, points) :
                         unit_to_physical(s.prior, points)
    elseif s.vectorized
        s.prior(points)
    else
        reduce(hcat, (s.prior(points[:, i]) for i in axes(points, 2));
               init = zeros(s.n_dim, 0))
    end

    log_w = log_w .- logsumexp(log_w)
    return (phys isa AbstractMatrix ? permutedims(phys) : phys), log_w, log_l
end
