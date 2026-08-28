# The tier-2 acceptance suite: the reference problems from the Python tests/test_sampler.py,
# with their original tolerances, plus the worked examples from the Python documentation.
#
# These are the tests that decide whether the port is correct. Everything above this file
# checks a component; these check that the whole method produces the right evidence and the
# right posterior on problems whose answers are known analytically.

using LinearAlgebra, Random, Statistics, Distributions, StableRNGs
using LogExpFunctions: logsumexp

identity_prior(x) = x

# Weighted mean and covariance of a posterior returned as (n_samples, n_params).
function weighted_moments(points, log_w)
    w = exp.(log_w)
    w ./= sum(w)
    mu = vec(sum(points .* w, dims = 1))
    d = points .- mu'
    cov = (d .* w)' * d
    return mu, cov
end

@testset "sampler: 2-D Gaussian accuracy" begin
    mvn = MvNormal([0.5, 0.5], 0.01 * I(2))
    s = Sampler(identity_prior, x -> logpdf(mvn, x); n_dim = 2, n_live = 500, seed = 0)
    run!(s; f_live = 0.1, n_eff = 10_000)

    @test abs(log_z(s)) < 0.05
    points, log_w, _ = posterior(s)
    mu, cov = weighted_moments(points, log_w)
    @test all(abs.(mu .- 0.5) .< 0.01)
    @test all(abs.(cov .- 0.01 * I(2)) .< 0.001)

    # Every bound must contain all later shells: the occupation matrix is lower-triangular.
    occ = shell_bound_occupation(s)
    @test all(occ[i, k] == 1.0 for i in axes(occ, 1), k in axes(occ, 2) if i >= k)
end

@testset "sampler: constant likelihood" begin
    s = Sampler(identity_prior, _ -> 0.0; n_dim = 2, n_live = 500, seed = 0)
    run!(s; f_live = 0.1, n_eff = 0)
    @test isapprox(log_z(s), 0.0; atol = 1e-8)
    @test length(s.bounds) == 1          # nothing to zoom in on, so no bound is ever added
end

@testset "sampler: enlarge_per_dim" begin
    # An enlargement this large makes every bound identical, so with no networks only one
    # bound survives and the sample is essentially uniform.
    like = x -> -sum(abs2, x .- 0.5) * 0.001
    s = Sampler(identity_prior, like; n_dim = 2, enlarge_per_dim = 100, n_networks = 0, seed = 0)
    run!(s; f_live = 0.1, n_eff = 0)

    @test isapprox(s.n_like, n_eff(s); rtol = 0, atol = 1)
    @test length(s.bounds) == 1
    @test isapprox(log_z(s), -4 * 0.5^3 / 3 * 0.001; rtol = 0, atol = 1e-4)
end

@testset "sampler: funnel" begin
    like = function (x)
        logpdf(Normal(0.5, 0.1), x[1]) +
        logpdf(Normal(0.5, exp(20 * (x[1] - 0.5)) / 100), x[2])
    end

    # Truth by Monte Carlo: the likelihood is normalised, so the evidence is the fraction of
    # its mass falling inside the unit cube.
    rng = StableRNG(0)
    x0 = 0.5 .+ 0.1 .* randn(rng, 1_000_000)
    x1 = 0.5 .+ (exp.(20 .* (x0 .- 0.5)) ./ 100) .* randn(rng, 1_000_000)
    log_z_true = log(mean(@. (0 < x0 < 1) & (0 < x1 < 1)))

    s = Sampler(identity_prior, like; n_dim = 2, n_networks = 1, seed = 0)
    run!(s)
    @test isapprox(log_z(s), log_z_true; rtol = 0, atol = 0.1)
end

@testset "sampler: plateau below threshold" begin
    like = x -> x[1] < 0.9 ? -Inf : log(x[1] - 0.9)
    log_z_true = log(0.5 * 0.1^2)
    for seed in 0:2
        s = Sampler(identity_prior, like; n_dim = 2, n_live = 1000, n_networks = 1, seed)
        run!(s; f_live = 0.1)
        @test isapprox(log_z(s), log_z_true; rtol = 0, atol = 0.1)
    end
end

@testset "sampler: staircase plateaus" begin
    like = x -> ceil(-log10(1 - x[1]))
    log_z_true = log(sum(0.9 .* 0.1 .^ (0:99) .* exp.(1 .+ (0:99))))

    s = Sampler(identity_prior, like; n_dim = 2, n_live = 2000, n_networks = 1, seed = 0)
    run!(s; f_live = 1e-6)
    @test isapprox(log_z(s), log_z_true; atol = 0.1)
    # Each bound must sit exactly on one plateau edge.
    @test s.shell_log_l_min[2:end] ≈ collect(2.0:(length(s.bounds)))
end

# ---------------------------------------------------------------------------------------------
# Tier 2b: the worked examples from the Python documentation, which publish their own numbers.
# ---------------------------------------------------------------------------------------------

@testset "docs example: README 3-D Gaussian" begin
    prior = Prior()
    for k in (:a, :b, :c)
        add_parameter!(prior, k)
    end
    mvn = MvNormal([0.4, 0.5, 0.6], 0.01 * I(3))
    s = Sampler(prior, d -> logpdf(mvn, [d[:a], d[:b], d[:c]]); n_live = 1000, seed = 0)
    run!(s)
    # Analytic: the Gaussian sits ~4 sigma inside the unit cube, so log Z = -6.4e-5.
    @test isapprox(log_z(s), -6.4e-5; atol = 0.02)
end

@testset "docs example: quickstart correlated Gaussian" begin
    # docs/guides/quickstart.rst, which reports log Z = -6.34. Analytic value is -6.3288.
    prior = Prior()
    add_parameter!(prior, :a, (-5, 5))
    add_parameter!(prior, :b, (-5, 5))
    add_parameter!(prior, :c, Normal(0, 2))
    cov = [1.0 0.0 0.9; 0.0 1.0 0.0; 0.9 0.0 1.0]
    mvn = MvNormal(zeros(3), cov)

    s = Sampler(prior, d -> logpdf(mvn, [d[:a], d[:b], d[:c]]); n_live = 1000, seed = 0)
    run!(s)
    @test isapprox(log_z(s), -6.3288; atol = 0.05)

    points, log_w, _ = posterior(s)
    mu, cv = weighted_moments(points, log_w)
    @test all(abs.(mu) .< 0.05)                       # all three means are zero

    # The posterior is Gaussian with precision inv(cov_L) + diag(0, 0, 1/4), the last term
    # being the N(0, 2) prior on c. Compare against that rather than against the likelihood's
    # own 0.9, which the prior shrinks.
    post_cov = inv(inv(cov) + Diagonal([0.0, 0.0, 1 / 4]))
    want = post_cov[1, 3] / sqrt(post_cov[1, 1] * post_cov[3, 3])
    @test isapprox(cv[1, 3] / sqrt(cv[1, 1] * cv[3, 3]), want; atol = 0.02)
end

@testset "docs example: prior-absorption identity" begin
    # docs/guides/priors.rst computes the same evidence two ways and requires them to agree.
    # The identity holds whether or not either number is right, so it isolates the prior
    # transform. Analytic value is -7.5523.
    mvn = MvNormal([1.5, 0.5, 1.5], 0.01 * I(3))
    dists = (Uniform(1, 3), Normal(2.0, 0.5), Gamma(1.0, 0.5))
    keys3 = (:a, :b, :c)

    prior = Prior()
    for (k, d) in zip(keys3, dists)
        add_parameter!(prior, k, d)
    end
    like = d -> logpdf(mvn, [d[:a], d[:b], d[:c]])
    s1 = Sampler(prior, like; n_live = 1000, seed = 0)
    run!(s1)

    # Now the same integral over a flat box, with the prior folded into the likelihood.
    flat = Prior()
    add_parameter!(flat, :a, (1, 3))
    add_parameter!(flat, :b, (0, 4))
    add_parameter!(flat, :c, (0, 4))
    log_prior = d -> sum(logpdf(dd, d[k]) for (k, dd) in zip(keys3, dists))

    s2 = Sampler(flat, log_prior; n_live = 1000, seed = 0)
    run!(s2)
    s3 = Sampler(flat, d -> like(d) + log_prior(d); n_live = 1000, seed = 0)
    run!(s3)

    @test isapprox(log_z(s1), -7.5523; atol = 0.05)
    @test isapprox(log_z(s3) - log_z(s2), log_z(s1); atol = 0.1)
end

@testset "sampler: threaded matches serial" begin
    # A deliberately slow, thread-safe likelihood so the parallel path is exercised.
    mvn = MvNormal([0.5, 0.5], 0.01 * I(2))
    like = x -> logpdf(mvn, x)

    a = Sampler(identity_prior, like; n_dim = 2, n_live = 400, seed = 1)
    run!(a; f_live = 0.1, n_eff = 2000)
    b = Sampler(identity_prior, like; n_dim = 2, n_live = 400, seed = 1, threaded = true)
    run!(b; f_live = 0.1, n_eff = 2000)

    # Threading changes the RNG partitioning, so the runs are not identical -- but both must
    # land on the same answer well inside their own uncertainty, ~1/sqrt(n_eff).
    @test isapprox(log_z(a), log_z(b); atol = 0.1)
    @test abs(log_z(b)) < 0.05
end

@testset "live parameter display" begin
    prior = Prior()
    add_parameter!(prior, :radius, (0.0, 10.0))
    add_parameter!(prior, :flux, Normal(0, 2))
    mvn = MvNormal([6.2, 1.0], Diagonal([0.05, 0.05]))
    s = Sampler(prior, d -> logpdf(mvn, [d[:radius], d[:flux]]);
                n_live = 300, n_networks = 1, seed = 0)
    run!(s; f_live = 0.1, n_eff = 500)

    out = sprint(io -> Nautilus.print_live_display(s; io))
    lines = split(strip(out), '\n')
    @test length(lines) == 2                       # one bar per free parameter
    @test all(occursin("|", l) for l in lines)
    @test occursin("radius", lines[1]) && occursin("flux", lines[2])

    # The bars must be drawn against the PRIOR range, and the two priors differ: uniform
    # (0, 10) and Normal(0, 2), whose 0.001/0.999 quantiles are about +/-6.18.
    @test occursin("+9.990", lines[1])
    @test occursin("+6.180", lines[2])

    # The live set has to sit where the likelihood is, not spread across the prior.
    @test occursin("*", lines[1])
    @test count(==('*'), lines[1]) < 25             # narrowed, not filling the bar

    # Returns the line count so a caller can redraw in place.
    @test Nautilus.print_live_display(s; io = IOBuffer()) == 2

    # Before any samples exist there is nothing to draw, and it must not throw.
    s0 = Sampler(x -> x, x -> 0.0; n_dim = 2, n_live = 50, seed = 0)
    @test Nautilus.print_live_display(s0; io = IOBuffer()) == 0
end

@testset "sampler: one dimension" begin
    # The Python refuses n_dim < 2 outright. That guard is inherited, not algorithmic, and is
    # not reproduced here — see the note in the Sampler constructor.
    @test_throws ArgumentError Sampler(identity_prior, x -> 0.0; n_dim = 0)

    # A Gaussian well inside the unit interval: Z = 1 - 2Φ(-5), so log Z is -5.7e-7.
    d = Normal(0.5, 0.1)
    for seed in 0:2
        s = Sampler(identity_prior, x -> logpdf(d, x[1]);
                    n_dim = 1, n_live = 400, n_networks = 1, seed)
        @test run!(s; f_live = 0.05, n_eff = 2000)
        @test isapprox(log_z(s), 0.0; atol = 0.05)

        points, log_w, _ = posterior(s)
        @test size(points, 2) == 1
        w = exp.(log_w); w ./= sum(w)
        mu = sum(points[:, 1] .* w)
        @test isapprox(mu, 0.5; atol = 0.01)
        @test isapprox(sqrt(sum(w .* (points[:, 1] .- mu) .^ 2)), 0.1; atol = 0.01)
    end

    # Two narrow, well-separated peaks: the ellipsoid union has to split in 1-D too, and the
    # evidence is still exactly 1 because the mixture is normalised and sits inside the range.
    like = x -> logsumexp([logpdf(Normal(0.25, 0.01), x[1]),
                           logpdf(Normal(0.75, 0.01), x[1])]) - log(2)
    s = Sampler(identity_prior, like; n_dim = 1, n_live = 500, n_networks = 1, seed = 0)
    run!(s; f_live = 0.02, n_eff = 2000)
    @test isapprox(log_z(s), 0.0; atol = 0.1)

    # Both peaks must be represented, not just the one the sampler happened to find first.
    points, log_w, _ = posterior(s)
    w = exp.(log_w); w ./= sum(w)
    lo = sum(w[points[:, 1] .< 0.5])
    @test 0.35 < lo < 0.65

    # And with the network disabled, i.e. plain multi-ellipsoidal sampling in 1-D.
    s0 = Sampler(identity_prior, x -> logpdf(d, x[1]);
                 n_dim = 1, n_live = 400, n_networks = 0, seed = 0)
    run!(s0; f_live = 0.05, n_eff = 1000)
    @test isapprox(log_z(s0), 0.0; atol = 0.05)
end
