# Precompilation workload.
#
# Time-to-first-fit was measured at 22 s in a fresh session against ~2 s for the same call
# immediately after, so essentially all of that gap is compilation. This runs a miniature
# version of a real fit at package build time so the machine code is cached into the package
# image instead of being generated in the user's first call.
#
# WHAT THIS CAN AND CANNOT CACHE. `Sampler{P,L}` is parameterised on the prior and likelihood
# function types, and a user's closure is a type that does not exist when this package is
# built. Every method specialised on it — the `run!` loop, `evaluate_likelihood`, `posterior`
# — must therefore still compile in the user's session, and no workload here can prevent that.
# What it does cache is everything reached with concrete, user-independent types: the bounds
# and their geometry, the MVEE, the Gaussian mixture, the MLP and the emulator. That is the
# larger half of the surface, and it is all reached through calls that are cheap to run.
#
# The sizes below are deliberately tiny. Julia specialises on types, not values, so a 3-column
# ellipsoid compiles the same code as a 3000-column one, and the hidden-layer sizes are runtime
# values inside `Vector{Matrix{Float64}}` rather than type parameters — `(4, 3, 2)` and
# `(100, 50, 20)` are both `NTuple{3,Int}` and share every specialisation. Keeping them small
# is what keeps this workload to a couple of seconds of build time.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    rng = Xoshiro(0)
    pts = rand(rng, 3, 80)
    log_l = -vec(sum(abs2, pts .- 0.5, dims = 1))
    log_l_min = median(log_l)
    tiny = (4, 3, 2)

    @compile_workload begin
        # --- geometry -------------------------------------------------------------------
        cube = UnitCube(3; rng = Xoshiro(0))
        in_bound(cube, pts); sample(cube, 4); log_volume(cube)

        ell = Ellipsoid(pts; rng = Xoshiro(0))
        in_bound(ell, pts); sample(ell, 4); log_volume(ell)
        transform_inverse(ell, transform(ell, pts))

        mix = UnitCubeEllipsoidMixture(pts; rng = Xoshiro(0))
        in_bound(mix, pts); sample(mix, 4); log_volume(mix); transform(mix, pts)

        # Both bound types the unions are instantiated with, and both split paths.
        ue = UnionBound(pts; bound_class = Ellipsoid, rng = Xoshiro(0))
        split!(ue; allow_overlap = false); sample(ue, 4); log_volume(ue); trim!(ue)
        um = UnionBound(pts; bound_class = UnitCubeEllipsoidMixture, rng = Xoshiro(0))
        split!(um); sample(um, 4); log_volume(um); in_bound(um, pts)

        # --- Gaussian mixture -----------------------------------------------------------
        gmm = fit_gaussian_mixture(pts, 2; rng = Xoshiro(0), n_init = 1)
        component_log_prob(gmm, pts)

        # --- network --------------------------------------------------------------------
        net = MLPRegressor(3, tiny, 1, Xoshiro(0))
        fit!(net, pts, log_l; rng = Xoshiro(0), max_iter = 2, batch_size = 32)
        predict(net, pts)
        emu = train_emulator(pts, log_l; n_networks = 1, hidden = tiny,
                             max_iter = 2, batch_size = 32)
        predict(emu, pts)

        nb = NeuralBound(pts, log_l, log_l_min; n_networks = 1, rng = Xoshiro(0),
                         hidden = tiny, max_iter = 2, batch_size = 32)
        in_bound(nb, pts)

        cb = NautilusBound(pts, log_l, log_l_min, log(0.5); n_networks = 1,
                           rng = Xoshiro(0), hidden = tiny, max_iter = 2, batch_size = 32)
        in_bound(cb, pts); sample(cb, 4); log_volume(cb); n_ell(cb); n_net(cb)

        # --- prior ----------------------------------------------------------------------
        pr = Prior()
        add_parameter!(pr, :a); add_parameter!(pr, :b, (-1, 1))
        add_parameter!(pr, :c, 0.0); add_parameter!(pr, :d, :a)
        u = rand(Xoshiro(0), 2, 5)
        unit_to_physical(pr, u); unit_to_dictionary(pr, u)

        # --- one end-to-end run ---------------------------------------------------------
        # Its `Sampler{P,L}` specialisations are useless to a caller with a different
        # closure, but the run drives every type-independent path in the sampler with real
        # data, which is what most of the cacheable work is reached through.
        s = Sampler(identity, x -> -sum(abs2, x .- 0.5) * 10;
                    n_dim = 2, n_live = 60, n_networks = 1, seed = 0)
        run!(s; f_live = 0.5, n_eff = 10, n_like_max = 400)
        posterior(s)
        posterior(s; equal_weight = true)
        log_z(s); n_eff(s); eta(s); shell_bound_occupation(s)
    end
end
