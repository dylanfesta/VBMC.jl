
using  Revise
using VBMC ; const VV = VBMC
using Distributions, Statistics , StatsBase , LinearAlgebra , Random
using BenchmarkTools
##
# test_logfun(x) = logpdf(MultivariateNormal( diagm(0=>[1.,1.]) ),x )

test_distr1D = Exponential(1.222)
test_distr1D = LogNormal(2.0,0.5)
test_logfun(x::Vector{Float64}) = logpdf(test_distr1D,x[1])


@btime _test = VV.slicesamplebnd( test_logfun , [0.2 ], 30_000)
_test = VV.slicesamplebnd( test_logfun , [0.2 ], 20_000)

using Plots
histogram(_test[:]; nbins=100, normed=true)
plot!(x->pdf(test_distr1D,x) , LinRange(0,30,500), linewidth=4 )

##
function test_logfun(x::Vector{Float64})
  x=x[1]
  if x<0.0 || x > 10.0
    -Inf
  elseif (x > 3.0 && x < 5. ) || x > 9
    log(3.)
  else
    log(1.)
  end
end


_test = VV.slicesamplebnd( test_logfun , [0.2 ], 10_000  )

using Plots
histogram(_test[:]; nbins=80, normed=true)
histogram!(_test2[:]; nbins=80, normed=true)
plot!(x->pdf(test_distr1D,x) , LinRange(0,15,100), linewidth=4 )

##
