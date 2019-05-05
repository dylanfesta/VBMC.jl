
#=
TODO
I think that this should become a separate module
=#

# core gaussian process type
# the hyperparameters are part of the mean function...
# what about the kernel ? mmmmh...
struct GaussianProcess{M<:AbstractMatrix,V<:AbstractVector}
  meanfun::GPLMeanFun
  X::M
  y::V
end

#####
##
# for symplicity hyperparameters are a vector
# make more GPs if you want more samples!
function GaussianProcessLite(meanfun::GPLMeanFun,
    X::AbstractMatrix,y::AbstractArray,
    hyp::AbstractVector)

  (d,n) = size(X)
  nhyp = length(hyp)
  ncov = d+1
  nhyp_meanf = length(meanfun)
  @assert(nhyp == nhyp_meanf + ncov + 1,
    "Number of hyperparameters mismatched with dimension of training inputs." )

  # extract hyperparamters
  ell = @. exp(hyp[1:d])
  sf2 = exp(2hyp[d+1])
  sn2 = exp(2hyp[ncov+1])
  sn2_mult = 1. # Effective noise variance multiplier
  # evaluate mean function on training inputs
  hyp_mean = hyp[ncov+2:ncov+1+nhyp_meanf] # Get mean function hyperparameters
  m = meanfun(X,hyp_mean) # compute the mean function

  # Compute kernel matrix K_mat
  # K_mat = sq_dist( Diagonal(inv.(ell))*(X') )
  @. K_mat = sf2 * exp(-K_mat/2)

  if sn2 < 1e-6   # Different representation depending on noise size
    for iter in 1:10     # Cholesky decomposition until it works
      [L,p] = chol( K_mat +  (sn2*sn2_mult*I) )
      p <= 0 &&  break # if p <= 0 break, else...
      sn2_mult *= 10.
    end
    sl = 1;
    pL = -L\(L'\eye(N));    % L = -inv(K+inv(sW^2))
    Lchol = 0;         % Tiny noise representation
  else
    for iter in 1:10
      [L,p] = chol(K_mat ./ (sn2*sn2_mult)+ I)
      p <=0 && break
      sn2_mult *= sn2_mult*10
    end
    sl = sn2*sn2_mult
    pL = L               # L = chol(eye(n)+sW*sW'.*K)
    Lchol = 1
  end
  #alpha = inv(K_mat + sn2.*eye(N)) * (y - m)
  alpha = ( L \ (L' \ (y-m) ) ) ./ sl;
  #
  # GP posterior parameters
  gp.post(s).alpha = alpha;
  gp.post(s).sW = ones(N,1)/sqrt(sn2*sn2_mult);   % sqrt of noise precision vector
  gp.post(s).L = pL;
  gp.post(s).sn2_mult = sn2_mult;
  gp.post(s).Lchol = Lchol;
  return ....  # TODO

end


#=

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
#
# constructor!
# this is gplite_post , does not include the update
function GaussianProcessLite(
    meanfun::GPLMeanFun,
    X::AbstractMatrix,
    y::AbstractVector,
    hyp_cov::AbstractVector,
    hprior)
  d,n = size(X) # training points and dimensions
  tol = 1E-6
  X_prior,y_prior = copy(X,y)
  ncov = d+1 # covariance function hyperparamters
  nhyp_meanf = length(meanfun)
  # hyperparameter are already in the meanfunction
  nhyp = ncov + nhyp_meanf+1
  hyp0 = fill(0., nhyp)
  LB = fill(NaN,nhyp)
  UB = fill(NaN,nhyp)
  hpri_mu = fill(NaN,nhyp)
  hpri_sigma = fill(NaN,nhyp)
  hpri_df = fill(7,nhyp)
  width = (-)(extrema(X_prior)...)
  height = (-)(extrema(y_prior)...)

  # hyp bounds
  LB_ell = LB(1:d)
  LB_ell[isnan.(LB_ell)] .=  log(width)+log(ToL)
  LB_sf = LB(d+1); if isnan(LB_sf); LB_sf = log(height)+log(ToL); end
  LB_sn = LB(ncov+1);     if isnan(LB_sn); LB_sn = log(ToL); end

  # mean function hyperparameter lower bounds
  LB_mean = LB[ncov+2:d+2+nmean]
  let idx = isnan.(LB_mean)
    LB_mean[idx] = meanfun.LB[idx]
  end
  UB_ell = UB[1:D]
  UB_ell[isnan.(UB_ell)] .= log(10width)
  UB_sf = UB[d+1];  if isnan(UB_sf); UB_sf = log(10height); end
  UB_sn = UB[ncov+1]; if isnan(UB_sn); UB_sn = log(height); end

  # Set mean function hyperparameters upper bounds
  UB_mean = UB[ncov+2:d+2+nmean];
  let idx = isnan(UB_mean)
    UB_mean[idx] = meanfun.UB[idx]
  end

  # Create lower and upper bounds
  LB = vcat(LB_ell,LB_sf,LB_sn,LB_mean)
  UB = vcat(UB_ell,UB_sf,UB_sn,UB_mean)
  UB = max.(LB,UB)

  # Plausible bounds for generation of starting points
  PLB_ell = log(width)+0.5log(ToL)
  PUB_ell = log(width)

  PLB_sf = log(height)+0.5log(ToL)
  PUB_sf = log(height)

  PLB_sn = 0.5log(ToL);
  PUB_sn = log(std(y_prior))

  PLB_mean = meanfun.PLB
  PUB_mean = meanfun.PUB

  PLB = vcat(PLB_ell,PLB_sf,PLB_sn,PLB_mean)
  PUB = vcat(PUB_ell,PUB_sf,PUB_sn,PUB_mean)

  PLB = min.(max.(PLB,LB),UB)
  PUB = max.(min.(PUB,UB),LB)

  gptrain_options = optimoptions('fmincon','GradObj','on','Display','off');


  hyp = zeros(Nhyp,Nopts);
  nll = [];


  =#
