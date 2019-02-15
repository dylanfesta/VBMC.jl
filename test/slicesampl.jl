
using  Revise
using VBMC ; const VV = VBMC
using Distributions, Statistics , StatsBase , LinearAlgebra , Random

##
# test_logfun(x) = logpdf(MultivariateNormal( diagm(0=>[1.,1.]) ),x )

test_distr1D = Exponential(1.222)
test_distr1D = LogNormal(0.3,1.34)
test_logfun(x::Vector{Float64}) = logpdf(test_distr1D,x[1])


_test = VV.slicesamplebnd( test_logfun , [0.2 ], 10000  )

using Plots
histogram(_test[:]; nbins=80, normed=true)
plot!(x->pdf(test_distr1D,x) , LinRange(0,15,100), linewidth=4 )

##
#=

function slicesamplebnd(logpdf::Function , x0::XT , nsampl::Integer ;
  logprior::Union{Function,Nothing} = nothing ,
  thinning::Integer = 1 ,
  burning::Union{Integer,Nothing} = nothing,
  step_out = false, display=true,
  widths = nothing ,
  boundaries=(-Inf,Inf),
  dostepout = false ,
  isadaptive = true ) where XT<:Union{AbstractArray{<:Real},Real}


=#
