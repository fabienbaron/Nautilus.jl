# Training behaviour of the hand-rolled MLP and the ensemble emulator.
#
# The acceptance criterion is the Python suite's (tests/test_neural.py): the emulator must
# predict the rank-transformed target to better than 0.3 * std(y).

using LinearAlgebra, Random, Statistics, StableRNGs
using StatsBase: ordinalrank, tiedrank
using Nautilus: MLPRegressor, fit!, predict, train_emulator, NeuralBound, polyval_at,
                in_bound, _batch_ranges

@testset "batch ranges match sklearn gen_batches" begin
    @test _batch_ranges(1000, 200) == [1:200, 201:400, 401:600, 601:800, 801:1000]
    @test _batch_ranges(950, 200) == [1:200, 201:400, 401:600, 601:800, 801:950]
    @test _batch_ranges(150, 200) == [1:150]
end

@testset "polyval_at" begin
    x = collect(range(-2, 2, length = 50))
    y = @. 1.0 + 2x - 0.5x^2 + 0.25x^3
    @test polyval_at(x, y, 3, 1.7) ≈ 1.0 + 2 * 1.7 - 0.5 * 1.7^2 + 0.25 * 1.7^3 rtol = 1e-8
end

@testset "MLP trains" begin
    rng = StableRNG(0)
    n_dim, n_points = 5, 1000
    x = rand(rng, n_dim, n_points)
    y = (ordinalrank(vec(sqrt.(sum(abs2, x .- 0.5, dims = 1))))) ./ n_points

    net = MLPRegressor(n_dim, (100, 50, 20), 1, Xoshiro(0))
    before = sqrt(mean(abs2, y .- predict(net, x)))
    fit!(net, x, y; rng = Xoshiro(0))
    after = sqrt(mean(abs2, y .- predict(net, x)))

    @test after < before
    @test after < 0.3 * std(y)          # tests/test_neural.py's criterion
end

@testset "emulator ensemble" begin
    rng = StableRNG(0)
    n_dim, n_points = 5, 1000
    x = rand(rng, n_dim, n_points)
    y = (ordinalrank(vec(sqrt.(sum(abs2, x .- 0.5, dims = 1))))) ./ n_points

    emu = train_emulator(x, y; n_networks = 1)
    @test sqrt(mean(abs2, y .- predict(emu, x))) < 0.3 * std(y)

    # An ensemble averages away some of the single-network noise, so it should not be worse.
    emu4 = train_emulator(x, y; n_networks = 4)
    @test length(emu4.networks) == 4
    @test sqrt(mean(abs2, y .- predict(emu4, x))) < 0.3 * std(y)

    # Seeds are fixed at 0..n-1, so training twice gives the identical ensemble.
    @test predict(emu4, x) == predict(train_emulator(x, y; n_networks = 4), x)
end

@testset "NeuralBound contains" begin
    # Mirrors tests/test_bounds.py::test_neural_bound_contains.
    rng = StableRNG(0)
    points = rand(rng, 4, 500)
    log_l = -vec(sqrt.(sum(abs2, points .- 0.5, dims = 1)))
    log_l_min = median(log_l)
    nb = NeuralBound(points, log_l, log_l_min; n_networks = 1, rng = StableRNG(0))

    test_points = rand(rng, 4, 100)
    test_log_l = -vec(sqrt.(sum(abs2, test_points .- 0.5, dims = 1)))
    inb = in_bound(nb, test_points)
    @test mean(test_log_l[inb] .> log_l_min) >= 0.9

    # With no network it degenerates to the outer ellipsoid.
    nb0 = NeuralBound(points, log_l, log_l_min; n_networks = 0, rng = StableRNG(0))
    @test nb0.emulator === nothing
    @test in_bound(nb0, test_points) == in_bound(nb0.outer_bound, test_points)
end
