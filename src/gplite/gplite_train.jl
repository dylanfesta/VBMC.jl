
#=
TODO
I think that this should become a separate module
=#


struct GaussianProcessLite
  meanfun::GPLMeanFun
  X::Matrix{Float64}
  y::Vector{Float64}
  post::Vector{Float64}
  nopt::Integer
  ninit::Integer
  sampler::GPLSampler # includes things like nsamples, thin, burn, etc
  degT::Integer
  logp_start::Vector{Float64}
  hprior::Hyperprior
end


# it has all the defaults

function GaussianProcessLite(n_samples::Integer,x::AbstractVecOrMat,y::AbstractVector)
  meanfun = MeanFConst(x,y)
  GaussianProcessLite(n_samples, x,y,meanfun)
end
function GaussianProcessLite(n_samples::Integer ,
   x::AbstractVecOrMat,y::AbstractVector,meanfun::GPLMeanFun)

   return nothing
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
