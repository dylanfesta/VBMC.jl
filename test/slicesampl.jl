
using  Revise
using VBMC ; const VV = VBMC
using Distributions, Statistics , StatsBase , LinearAlgebra , Random


##

test_logfun(x) = logpdf(MultivariateNormal( diagm(0=>[1.,1.]) ),x )


VV.slicesamplebnd( test_logfun , [0. , 0 ], 100  )

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
