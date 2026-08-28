# Cross-checks against reference values dumped from the Python package by
# test/fixtures/dump_fixtures.py. These cover the paths that are deterministic given their
# input, so they are the hard correctness gate; anything stochastic is checked statistically
# elsewhere.
#
# HDF5 stores row-major and Julia reads column-major, so a Python (n_points, n_dim) array
# arrives here as (n_dim, n_points) with no transpose needed -- which is exactly this port's
# layout. The same applies to sklearn's coefs_, stored (fan_in, fan_out) and read as
# (fan_out, fan_in), matching MLPRegressor.W.

using HDF5, LinearAlgebra, Random, Statistics, StableRNGs
using StatsBase: tiedrank
using Nautilus: minimum_volume_enclosing_ellipsoid, ellipsoids_overlap, MLPRegressor,
                predict, sample

const FIXTURES = joinpath(@__DIR__, "fixtures", "fixtures.h5")

allclose_f(a, b; rtol = 1e-5, atol = 1e-8) = all(@. abs(a - b) <= atol + rtol * abs(b))

if !isfile(FIXTURES)
    @warn "fixtures.h5 missing; run test/fixtures/dump_fixtures.py to enable cross-checks"
else
    h5open(FIXTURES) do f

        @testset "fixture: MVEE vs Python" begin
            pts = read(f["mvee"]["points"])
            @test size(pts, 1) == 10                    # confirms the layout assumption
            c, A = minimum_volume_enclosing_ellipsoid(pts)[1:2]
            c_py, A_py = read(f["mvee"]["c"]), read(f["mvee"]["A"])

            # Not bit-identical: the batch ordering and the symmetrisation differ. Both are
            # approximations to the same ellipsoid, so they are compared against each other
            # at the accuracy the algorithm itself delivers.
            @test allclose_f(c, c_py; rtol = 0, atol = 1e-3)
            @test allclose_f(A, A_py; rtol = 0, atol = 2e-2)
        end

        @testset "fixture: rank score vs Python" begin
            log_l = read(f["score"]["log_l"])
            log_l_min = read_attribute(f["score"], "log_l_min")
            score_py = read(f["score"]["score"])

            above = log_l .>= log_l_min
            n_above, n_below = count(above), count(!, above)
            score = zeros(length(log_l))
            score[above] .= 0.5 .* (1 .+ (tiedrank(log_l[above]) .- 0.5) ./ n_above)
            score[.!above] .= 0.5 .* ((tiedrank(log_l[.!above]) .- 0.5) ./ n_below)

            # Deterministic, ties included: this must be exact.
            @test allclose_f(score, score_py; rtol = 0, atol = 1e-12)
        end

        @testset "fixture: ellipsoids_overlap vs Python" begin
            base = read(f["overlap"]["base"])
            shift = read(f["overlap"]["shift"])
            want = read(f["overlap"]["overlap"]) .== 1

            got = map(eachindex(shift)) do i
                e1 = Ellipsoid(base; rng = StableRNG(0))
                e2 = Ellipsoid(base .+ shift[i]; rng = StableRNG(0))
                ellipsoids_overlap([e1, e2])
            end
            # A different minimiser, but it must reach the same verdict on every pair.
            @test got == want
        end

        @testset "fixture: mixture dimension selection vs Python" begin
            pts = read(f["mixture"]["points"])
            want = read(f["mixture"]["dim_cube"]) .== 1
            b = UnitCubeEllipsoidMixture(pts; rng = StableRNG(0))
            @test b.dim_cube == want
            @test log_volume(b) ≈ read_attribute(f["mixture"], "log_v") rtol = 1e-3
        end

        @testset "fixture: MLP forward pass vs Python" begin
            g = f["mlp"]
            x = read(g["x"])
            pred_py = read(g["pred"])
            L = read_attribute(g, "n_layers") - 1

            W = [read(g["coefs_$(k)"]) for k in 0:(L - 1)]
            b = [read(g["intercepts_$(k)"]) for k in 0:(L - 1)]
            @test size(W[1]) == (100, size(x, 1))       # confirms the coefs_ orientation

            net = MLPRegressor(W, b)
            # Same weights, same arithmetic: this is the one place that must be exact.
            @test allclose_f(predict(net, x), pred_py; rtol = 0, atol = 1e-10)
        end
    end
end
