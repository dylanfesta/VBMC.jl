module VBMC
using Distributions , LinearAlgebra , Statistics, StatsBase , Random

using Formatting


include("utils/sq_dist.jl")
include("utils/slicesampler.jl")
include("gplite/gplite_meanfun.jl")


end # module
