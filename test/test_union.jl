# Mirrors the Union half of the Python tests/test_bounds.py, plus direct cover for the
# Gaussian mixture and the ellipsoid-overlap predicate that replace scikit-learn and scipy.

using LinearAlgebra, Random, Statistics, StableRNGs
using Nautilus: fit_gaussian_mixture, component_log_prob, ellipsoids_overlap,
                golden_section_min, sample, nbounds

function hypersphere(rng, n_dim = 3, n_points = 1000)
    pts = randn(rng, n_dim, n_points)
    r = rand(rng, n_points)
    for i in 1:n_points
        col = @view pts[:, i]
        col ./= norm(col)
        col .*= r[i]^(1 / n_dim)
    end
    return pts
end

@testset "golden_section_min" begin
    x, f = golden_section_min(t -> (t - 0.3)^2 + 1.0, 1e-9, 1 - 1e-9)
    @test x ≈ 0.3 atol = 1e-6
    @test f ≈ 1.0 atol = 1e-9
end

@testset "fit_gaussian_mixture" begin
    rng = StableRNG(0)
    # Two well-separated blobs: EM must recover the means, weights and covariances.
    a = randn(rng, 2, 400) .* 0.3 .+ [0.0, 0.0]
    b = randn(rng, 2, 600) .* 0.3 .+ [8.0, 8.0]
    g = fit_gaussian_mixture(hcat(a, b), 2; rng = StableRNG(1))

    order = sortperm([m[1] for m in g.means])
    @test g.means[order[1]] ≈ [0.0, 0.0] atol = 0.1
    @test g.means[order[2]] ≈ [8.0, 8.0] atol = 0.1
    @test g.weights[order] ≈ [0.4, 0.6] atol = 0.05
    @test all(isapprox(g.covs[c], 0.09 * I(2); atol = 0.03) for c in 1:2)

    # The labels the split step derives from it must recover the true grouping.
    labels = [argmax(view(component_log_prob(g, hcat(a, b)), :, i)) for i in 1:1000]
    @test length(unique(labels[1:400])) == 1
    @test length(unique(labels[401:1000])) == 1
    @test labels[1] != labels[end]
end

@testset "ellipsoids_overlap" begin
    rng = StableRNG(0)
    blob = hypersphere(rng)
    near = [Ellipsoid(blob; rng = StableRNG(0)),
            Ellipsoid(blob .+ 0.5; rng = StableRNG(0))]
    far  = [Ellipsoid(blob; rng = StableRNG(0)),
            Ellipsoid(blob .+ 100; rng = StableRNG(0))]
    @test ellipsoids_overlap(near)
    @test !ellipsoids_overlap(far)
    @test !ellipsoids_overlap(near[1:1])
end

@testset "UnionBound construction" begin
    rng = StableRNG(0)
    @test_throws ArgumentError UnionBound(rand(rng, 10, 100); n_points_min = 5)
end

@testset "UnionBound split" begin
    blob = hypersphere(StableRNG(0))
    points = hcat(blob, blob .+ 100, blob .+ 101)
    u = UnionBound(points; enlarge_per_dim = 1.0 + 1e-9, unit = false, rng = StableRNG(0))

    # The clouds at +100 and +101 overlap each other, so without overlaps only two
    # ellipsoids are reachable.
    while split!(u; allow_overlap = false)
    end
    @test nbounds(u) == 2
    @test all(in_bound(u, points))

    # Allowing overlaps buys exactly one more.
    @test split!(u)
    @test !split!(u)
    @test nbounds(u) == 3
    @test all(in_bound(u, points))
end

@testset "UnionBound split terminates" begin
    x = collect(range(-1, 1, length = 30))
    points = permutedims(hcat(x, x .^ 2))
    u = UnionBound(points; rng = StableRNG(0))

    n = 0
    while split!(u)
        n += 1
    end
    @test n > 0
    @test all(size(p, 2) >= u.n_points_min for p in u.points_bounds)
end

@testset "UnionBound split and trim" begin
    blob = hypersphere(StableRNG(0))
    points = hcat(blob, blob .+ 10, blob[:, 1:30] .+ 1e7)
    u = UnionBound(points; unit = false, n_points_min = 50, rng = StableRNG(0))

    @test log_volume(u) > 15
    @test split!(u)
    @test split!(u)
    # Splitting cannot help much: one ellipsoid still spans the cloud at 1e7 and a handful
    # of points near the origin.
    @test log_volume(u) > 15
    @test trim!(u)
    @test log_volume(u) < 5
    # The survivors all have comparable density, so nothing more should go.
    @test !trim!(u)
end

@testset "UnionBound sample and contains" begin
    blob = hypersphere(StableRNG(0)) .+ 50
    u = UnionBound(blob; enlarge_per_dim = 1.0, unit = false, rng = StableRNG(0))
    for _ in 1:4
        split!(u)
    end

    points = sample(u, 100)
    @test size(points) == (3, 100)
    @test all(in_bound(u, points))

    # Splitting a uniformly-filled ball can only raise the total volume, so every split is
    # correctly refused and the union stays a single ellipsoid.
    @test nbounds(u) == 1

    # That single ellipsoid is the MVEE of a unit 3-ball, so its volume is known exactly.
    @test log_volume(u) ≈ log(4pi / 3) rtol = 0.01

    # A deliberately larger union around the drawn points must escape the original.
    u_large = UnionBound(points; enlarge_per_dim = 1.1, unit = false, rng = StableRNG(0))
    @test !all(in_bound(u, sample(u_large, 100)))
end

@testset "UnionBound rng" begin
    blob = hypersphere(StableRNG(0))
    mk = seed -> (u = UnionBound(blob; unit = false, rng = StableRNG(seed)); split!(u); u)
    u, u_same, u_diff = mk(0), mk(0), mk(1)
    points = sample(u, 100)
    @test points == sample(u_same, 100)
    @test points != sample(u_diff, 100)
    @test points != sample(u, 100)
end

@testset "UnionBound unit cube intersection" begin
    rng = StableRNG(0)
    pts = rand(rng, 3, 500)
    u = UnionBound(pts; rng = StableRNG(0))
    s = sample(u, 500)
    @test all(0 .<= s .< 1)            # the cube factor is enforced on the way out
    @test all(in_bound(u, pts))
end
