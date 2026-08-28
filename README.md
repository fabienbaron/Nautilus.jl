# Nautilus.jl

A Julia port of [nautilus](https://github.com/johannesulf/nautilus) — importance nested
sampling with neural-network boosted space exploration (Lange 2023,
[MNRAS 525, 3181](https://academic.oup.com/mnras/article/525/2/3181/7243406)).

```julia
using Nautilus, Distributions, LinearAlgebra

prior = Prior()
add_parameter!(prior, :a, (-5, 5))
add_parameter!(prior, :b, (-5, 5))
add_parameter!(prior, :c, Normal(0, 2))

mvn = MvNormal(zeros(3), [1.0 0.0 0.9; 0.0 1.0 0.0; 0.9 0.0 1.0])
sampler = Sampler(prior, d -> logpdf(mvn, [d[:a], d[:b], d[:c]]); n_live = 1000, seed = 0)
run!(sampler; verbose = true)

points, log_w, log_l = posterior(sampler)
log_z(sampler)          # -6.33; the Python package's own docs quote -6.34
```

## Status

The method is fully ported and validated: the geometric bounds, the multi-ellipsoid
decomposition, the neural likelihood emulator and the sampler. `n_networks = 0` falls back to
pure multi-ellipsoidal sampling.

Deliberately **not** ported: HDF5 checkpointing and resume, blobs, and periodic parameters.

Added beyond upstream:

- `live_display = true` draws a per-parameter bar showing the live set narrowing inside the
  prior range, in the spirit of UltraNest's live logger — the Python package prints only a
  single status line.
- One-dimensional problems run. The Python refuses them with a hard error; that guard is
  inherited from its v0.2.0 rewrite and is not algorithmic. (Nested sampling is still the wrong
  tool in 1-D — quadrature reaches machine precision for a fraction of the likelihood calls —
  but a one-parameter fit is a real thing to want.)

## Dependencies

Deliberately light. The two pieces of machinery the Python package takes from scikit-learn —
a two-component Gaussian mixture and the MLP emulator — are implemented here instead, so no
neural-network framework enters the dependency graph. Both are faster than the general-purpose
Julia alternatives at this problem size; see the comments in `src/gmm.jl` and `src/mlp.jl`.

## Validation

`Pkg.test()` runs three tiers:

- **Cross-checks against the Python package**, from `test/fixtures/fixtures.h5`: the MLP
  forward pass reproduces scikit-learn's own trained weights to 1e-10, rank scores to 1e-12,
  and every `ellipsoids_overlap` verdict agrees.
- **The Python suite's reference problems** at their original tolerances: 2-D Gaussian
  accuracy, Neal's funnel, both likelihood-plateau problems, the Gaussian shell, two peaks.
- **The worked examples from the Python documentation**, which publish their own numbers:
  quickstart (log Z = -6.3288) and the prior-absorption identity (-7.5523, computed two ways).

## Provenance

The port was initially generated in bulk by an LLM (Claude), then supervised and debugged by
Fabien Baron using [JET.jl](https://github.com/aviatesk/JET.jl) and
[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).

## Licence

MIT, as the original.
