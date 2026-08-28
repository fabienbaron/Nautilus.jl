using Test
using Nautilus

@testset "Nautilus" begin
    include("test_basic.jl")
    include("test_union.jl")
    include("test_mlp.jl")
    include("test_nautilus_bound.jl")
    include("test_sampler.jl")
    include("test_fixtures.jl")
end
