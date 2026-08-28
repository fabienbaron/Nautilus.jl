# Mirrors the basic-bound half of the Python tests/test_bounds.py, with points as columns.
#
# None of these need the Python package: the MVEE fixture has an analytically known answer
# (a unit sphere), and the ellipsoid volume has a closed form. That makes this file the
# correctness gate for src/bounds/basic.jl on its own terms.

using LinearAlgebra, Random, Statistics, StableRNGs
using SpecialFunctions: gamma
using Nautilus: minimum_volume_enclosing_ellipsoid, invert_spd, sample, reset!

# numpy's `np.allclose` is ELEMENT-WISE; Julia's `isapprox` on arrays compares Frobenius
# norms and so quietly tightens by roughly sqrt(n) on a large array. Every tolerance below is
# copied from the Python suite, so they need numpy's meaning.
allclose(a, b; rtol = 1e-5, atol = 1e-8) = all(@. abs(a - b) <= atol + rtol * abs(b))

# 2*n_dim points on the boundary of a unit hypersphere centred on 0.5, one per column.
function points_on_hypersphere_boundary(n_dim = 10)
    points = fill(0.5, n_dim, 2 * n_dim)
    for i in 0:(2 * n_dim - 1)
        points[i ÷ 2 + 1, i + 1] += isodd(i) ? 1 : -1
    end
    return points
end

function random_points_from_hypersphere(rng, n_dim = 3, n_points = 1000)
    pts = randn(rng, n_dim, n_points)
    r = rand(rng, n_points)
    for i in 1:n_points
        col = @view pts[:, i]
        col ./= norm(col)
        col .*= r[i]^(1 / n_dim)
    end
    return pts
end

@testset "invert_spd" begin
    rng = StableRNG(0)
    for n in (1, 2, 5, 10)
        L = randn(rng, n, n)
        m = L * L' + n * I
        @test allclose(invert_spd(m), inv(m))
    end
end

@testset "UnitCube" begin
    n_dim, n_points = 3, 200
    cube = UnitCube(n_dim; rng = StableRNG(0))
    points = sample(cube, n_points)
    @test size(points) == (n_dim, n_points)
    @test all(0 .<= points .< 1)
    @test all(in_bound(cube, points))
    @test log_volume(cube) == 0
end

@testset "UnitCube rng" begin
    a = sample(UnitCube(3; rng = StableRNG(0)), 100)
    @test a == sample(UnitCube(3; rng = StableRNG(0)), 100)
    @test a != sample(UnitCube(3; rng = StableRNG(1)), 100)
end

@testset "minimum_volume_enclosing_ellipsoid" begin
    base = points_on_hypersphere_boundary()
    c_true = vec(median(base, dims = 2))
    # One extra interior point, without which the algorithm is exact after one iteration.
    # 0.5488135039273248 is numpy's first `random()` under legacy seed 0, so this fixture is
    # the same point set the Python test uses.
    extra = c_true .+ (0.5488135039273248 - 0.5)
    points = hcat(base, extra)

    c, A = minimum_volume_enclosing_ellipsoid(points)[1:2]
    @test allclose(c, c_true; rtol = 0, atol = 1e-3)
    @test allclose(A, Matrix{Float64}(I, 10, 10); rtol = 0, atol = 1e-2)
end

@testset "Ellipsoid construction errors" begin
    rng = StableRNG(0)
    @test_throws ArgumentError Ellipsoid(rand(rng, 10, 10))              # n_points == n_dim
    @test_throws ArgumentError Ellipsoid(rand(rng, 10, 100); enlarge_per_dim = 0.9)
end

@testset "Ellipsoid sample and contains" begin
    base = points_on_hypersphere_boundary()
    ell = Ellipsoid(base; enlarge_per_dim = 1.0, rng = StableRNG(0))
    c = mean(base)

    points = sample(ell, 100)
    @test size(points) == (size(base, 1), 100)
    @test all(norm(points[:, i] .- c) < 1 + 1e-9 for i in 1:100)
    @test all(in_bound(ell, points))

    ell2 = Ellipsoid(points; enlarge_per_dim = 1.1, rng = StableRNG(0))
    points2 = sample(ell2, 100)
    @test !all(norm(points2[:, i] .- c) < 1 for i in 1:100)
    @test all(in_bound(ell2, points2))
end

@testset "Ellipsoid volume" begin
    base = points_on_hypersphere_boundary()
    n_dim = size(base, 1)
    for enlarge_per_dim in (1.0, 1.1, pi / 2)
        ell = Ellipsoid(base; enlarge_per_dim)
        @test log_volume(ell) ≈
              log(enlarge_per_dim^n_dim * pi^(n_dim / 2) / gamma(n_dim / 2 + 1)) rtol = 1e-5
    end
end

@testset "Ellipsoid transform" begin
    pts = random_points_from_hypersphere(StableRNG(0))
    ell = Ellipsoid(pts; rng = StableRNG(0))
    points = sample(ell, 100)
    points_t = transform(ell, points)
    @test all(abs.(points_t) .< 1 + 1e-9)
    @test allclose(transform_inverse(ell, points_t), points)
end

@testset "Ellipsoid rng" begin
    pts = random_points_from_hypersphere(StableRNG(0))
    ell      = Ellipsoid(pts; rng = StableRNG(0))
    ell_same = Ellipsoid(pts; rng = StableRNG(0))
    ell_diff = Ellipsoid(pts; rng = StableRNG(1))
    points = sample(ell, 1000)
    @test points == sample(ell_same, 1000)
    @test points != sample(ell_diff, 1000)
    @test points != sample(ell, 1000)          # the stream advances
end

@testset "UnitCubeEllipsoidMixture" begin
    rng = StableRNG(0)
    # Uniform points in a 4-D cube: every dimension should be handled by the cube, since no
    # ellipsoid beats the unit cube on a uniform fill.
    pts = rand(rng, 4, 500)
    b = UnitCubeEllipsoidMixture(pts; rng = StableRNG(0))
    @test all(b.dim_cube)
    @test b.ellipsoid === nothing
    @test log_volume(b) == 0
    @test all(in_bound(b, pts))

    s = sample(b, 200)
    @test size(s) == (4, 200)
    @test all(in_bound(b, s))

    # A distribution narrow in two dimensions and full-width in the others: the narrow ones
    # should go to the ellipsoid, the wide ones to the cube.
    rng = StableRNG(1)
    wide = rand(rng, 2, 1000)
    narrow = 0.5 .+ 0.01 .* randn(rng, 2, 1000)
    pts = vcat(narrow, wide)
    b = UnitCubeEllipsoidMixture(pts; rng = StableRNG(0))
    @test b.ellipsoid !== nothing
    @test log_volume(b) < 0
    @test all(in_bound(b, pts))
    @test count(b.dim_cube) >= 1

    t = transform(b, pts)
    @test all(abs.(t) .< 1 + 1e-9)
end
