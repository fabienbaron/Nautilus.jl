"""
    Nautilus

Importance nested sampling with neural-network boosted space exploration — a Julia port of
the `nautilus` Python package (Lange 2023, MNRAS 525, 3181).

Work in progress: the geometric bounds are in place, the sampler is not yet.
"""
module Nautilus

using LinearAlgebra
using Random
using Random: randperm
using Statistics
using Random: shuffle!
using LinearAlgebra: BLAS
using Printf

using SpecialFunctions: loggamma
using LogExpFunctions: logsumexp
using StatsBase: tiedrank
using Distributions
using Clustering

include("prior.jl")
include("gmm.jl")
include("mlp.jl")
include("emulator.jl")
include("bounds/basic.jl")
include("bounds/union.jl")
include("bounds/neural.jl")
include("bounds/nautilus.jl")
include("sampler.jl")
include("precompile.jl")

export Prior, add_parameter!, dimensionality, unit_to_physical,
       physical_to_dictionary, unit_to_dictionary
export UnitCube, Ellipsoid, UnitCubeEllipsoidMixture, UnionBound, NeuralBound, NautilusBound
export Sampler, run!, posterior, log_z, n_eff, eta, f_live,
       shell_bound_occupation, set_discard_exploration!
export in_bound, log_volume, n_ell, n_net, transform, transform_inverse, split!, trim!

end # module
