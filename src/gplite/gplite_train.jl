
#=
NB
I think that this should become a separate module
=#


struct GaussianProcessLite
  meanfun::GPLMeanFun
  X::Matrix{Float64}
  y::Vector{Float64}
  post::Vector{Float64}

end

# this is gplite_post , does not include the update

function GaussianProcessLite(meanfun::GPLMeanFun,X::AbstractMatrix ,y::AbstractVector)
  n,d = size(X)
  nhyp = length(meanfun)
