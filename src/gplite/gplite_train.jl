
#=
NB
I think that this should become a separate module
=#


struct GaussianProcessLite
  meanfun::GPLMeanFun
  X::Matrix{Float64}
  y::Vector{Float64}
  post::Vector{Float64}
  nopt::Integer
  ninit::Integer
  sampler::GLPSampler # includes things like nsamples, thin, burn, etc
  degT::Integer
  logp_start::Vector{Float64}
  hprior::Hyperprior
end

# this is gplite_post , does not include the update

function GaussianProcessLite(
    meanfun::GPLMeanFun,
    X::AbstractMatrix,
    y::AbstractVector,
    hyp_cov::AbstractVector,
    hprior)
  n,d = size(X)
  tol = 1E-6
  X_prior,y_prior = copy(X,y)
  ncov = d+1 # covariance function hyperparamters
  nhyp_meanf = length(meanfun)
  meaninfo = do_some_stuff(meanfun)
