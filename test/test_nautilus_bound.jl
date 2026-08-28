# Mirrors tests/test_bounds.py's NautilusBound tests. These are the phase-4 acceptance gate:
# the gaussian shell and two-peaks problems are where the neural bounds have to earn their
# keep, and both check the recovered volume against an analytically known value.

using LinearAlgebra, Random, Statistics, StableRNGs
using Nautilus: sample, n_ell, n_net, in_bound

dist_from(points, c) = vec(sqrt.(sum(abs2, points .- c, dims = 1)))

@testset "NautilusBound sample and contains" begin
    rng = StableRNG(0)
    points = rand(rng, 4, 500)
    log_l = -dist_from(points, 0.5)
    log_l_min = median(log_l)
    nb = NautilusBound(points, log_l, log_l_min, log(0.5); n_networks = 1, rng = StableRNG(0))

    test_points = rand(rng, 4, 100)
    inb = in_bound(nb, test_points)
    @test mean(-dist_from(test_points, 0.5)[inb] .> log_l_min) >= 0.9

    drawn = sample(nb, 100)
    @test mean(-dist_from(drawn, 0.5) .> log_l_min) >= 0.9
    @test all(in_bound(nb, drawn))
end

@testset "NautilusBound gaussian shell" begin
    radius, width = 0.45, 0.01
    rng = StableRNG(0)
    points = rand(rng, 2, 10000)
    log_l = -((dist_from(points, 0.5) .- radius) ./ width) .^ 2
    keep = log_l .> -100
    points, log_l = points[:, keep], log_l[keep]

    log_l_min = -1.0
    log_v_target = log(2pi * radius * width * 2)

    nb = NautilusBound(points, log_l, log_l_min, log_v_target;
                       split_threshold = 1, n_networks = 1, rng = StableRNG(0))
    drawn = sample(nb, 10000)
    drawn_log_l = -((dist_from(drawn, 0.5) .- radius) ./ width) .^ 2

    @test isapprox(log_volume(nb), log_v_target; rtol = 0, atol = log(2))
    @test mean(drawn_log_l .> log_l_min) > 0.5
    @test n_net(nb) == 1
end

@testset "NautilusBound two peaks" begin
    radius = 1e-5
    rng = StableRNG(0)
    points = hcat(randn(rng, 2, 1000) .* radius .+ 0.1,
                  randn(rng, 2, 1000) .* radius .+ 0.9)
    like = p -> -min.(dist_from(p, 0.1), dist_from(p, 0.9)) ./ radius

    log_l = like(points)
    log_l_min = -1.0
    log_v_target = log(2pi * radius^2)

    nb = NautilusBound(points, log_l, log_l_min, log_v_target;
                       n_networks = 1, rng = StableRNG(0))
    drawn = sample(nb, 10000)

    @test isapprox(log_volume(nb), log_v_target; rtol = 0, atol = 0.1)
    @test mean(like(drawn) .> log_l_min) > 0.9
    # The two peaks must be resolved into two separate ellipsoids, hence two networks.
    @test n_net(nb) == 2
    @test n_ell(nb) >= 1
end
