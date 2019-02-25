

abstract type GPLMeanFun end

struct GPZero <: GPLMeanFun end
struct GPConst <: GPLMeanFun
    c::Float64
end
struct GPLinear <: GPLMeanFun
  x0::Float64
  w::Vector{Float64}
end
struct GPQuad <: GPLMeanFun
  x0::Float64
  xm::Vector{Float64}
  xmsq::Vector{Float64}
end
struct GPNegQuad <: GPLMeanFun
  x0::Float64
  xm::Vector{Float64}
  omega::Vector{Float64}
end
struct GPPosQuad <: GPLMeanFun
  x0::Float64
  xm::Vector{Float64}
  omega::Vector{Float64}
end
struct GPSE <: GPLMeanFun
  x0::Float64
  h::Float64
  xm::Vector{Float64}
  omega::Vector{Float64}
end
struct GPNegSE <: GPLMeanFun
  x0::Float64
  h::Float64
  xm::Vector{Float64}
  omega::Vector{Float64}
end


"""
        nhyper(mfun::GPLMeanFun)
returns the number of hyperparameter of the given GPLite mean function
"""
function nhyper(mfun::GPLMeanFun)
    @error "nhyper not defined for this category of mean function!!!"
    nothing
end
function nhyper(mfun::GPZero)
    return 0
end
function nhyper(mfun::GPConst)
    return 1
end
function nhyper(mfun::GPLinear)
    return 1+length(mfun.w)
end
function nhyper(mfun::G) where G<:Union{GPQuad,GPNegQuad,GPPosQuad,GPSE,GPNegSE}
    return 1+2length(mfun.xm)
end
function nhyper(mfun::G) where G<:Union{GPSE,GPNegSE}
    return 2+2length(mfun.xm)
end

# auxiliary functions to test the sizes of all the elements
function _test_grad_size(grad::Matrix{Float64},n,meanfun::GPLMeanFun)
  nh = nhyper(meanfun)
  dg,ng = size(grad)
  @assert n == ng "grad matrix columns and input points do not match"
  @assert dg == nh " number of hyperparamters and grad matrix row do not match "
  nothing
end
function _test_grad_size(grad::Nothing,whatevs...)
  return nothing
end

function _test_Xgrad_size(X::Vector{Float64},grad,meanfun::GPLMeanFun)
  _test_grad_size(grad,length(X),meanfun)
end
function _test_Xgrad_size(X::Vector{Float64},grad,meanfun::GPLinear)
  n=length(X)
  @assert length(meanfun.w) == n  "issue with linear mean function! Not right size!"
  _test_grad_size(grad,n,meanfun)
end
function _test_Xgrad_size(X::Vector{Float64},
                grad,meanfun::G) where G<:Union{GPQuad,GPPosQuad,GPNegQuad}
  n=length(X)
  (a,b,c) = [ getfield(meanfun,nm) for nm in fieldnames(G) ]
  @assert length(b) == n  "issue with quadratic mean function! Not right size!"
  @assert length(c) == n  "issue with quadratic mean function! Not right size!"
  _test_grad_size(grad,n,meanfun)
end
function _test_Xgrad_size(X::Vector{Float64},
                grad,meanfun::G) where G<:Union{GPSE,GPNegSE}
  n=length(X)
  (a,b,c,d) = [ getfield(meanfun,nm) for nm in fieldnames(G) ]
  @assert length(c) == n  "issue with quadratic mean function! Not right size!"
  @assert length(d) == n  "issue with quadratic mean function! Not right size!"
  _test_grad_size(grad,n,meanfun)
end


# compute the mean functions in a series of points, with or without the gradients
function _get_mfun_grad(X::Vector{Float64},meanfun::GPLMeanFun,
                grad::T) where T<:Union{Nothing,AbstractVector}
  @error " This function has not been defined for $(typeof(meanfun)) ! "
  return nothing
end

function _get_mfun_grad(X::Vector{Float64},meanfun::GPZero,
                          grad::T) where T<:Union{Nothing,AbstractVector}
  n=length(X)
  _test_grad_size(grad,n,meanfun)
  if !isnothing(grad)
    fill!(grad,NaN)  # should it be 1 anyway?
  end
  zero(X)
end
function _get_mfun_grad(X::Vector{Float64},meanfun::GPConst,
                      grad::T) where T<:Union{Nothing,AbstractVector}
  n=length(X)
  _test_grad_size(grad,n,meanfun)
  if !isnothing(grad)
    fill!(grad,1.0)
  end
  fill(meanfun.x0,n)
end
function _get_mfun_grad(X::Vector{Float64},meanfun::GPLinear,
                            grad::T) where T<:Union{Nothing,AbstractVector}
  _test_Xgrad_size(grad,X,meanfun)
  x0, w = meanfun.x0 , meanfun.w
  if !isnothing(grad)
    grad .= vcat(1.0,X)
  end
  @. x0 + w*X
end
function _get_mfun_grad(X::Vector{Float64},meanfun::GPQuad,
                  grad::T) where T<:Union{Nothing,AbstractVector}
  _test_Xgrad_size(grad,X,meanfun)
  x0, xm , xmsq = meanfun.x0 , meanfun.xm , meanfun.xmsq
  Xsq = X.^2
  if !isnothing(grad)
    grad .= vcat(1.0, X , Xsq)
  end
  @. x0 + xm*X + xmsq*Xsq # broadcast magic!
end
function _get_mfun_grad(X::Vector{Float64},meanfun::GPPosQuad,
                  grad::T) where T<:Union{Nothing,AbstractVector}
  _test_Xgrad_size(grad,X,meanfun)
  x0, xm , omega = meanfun.x0 , meanfun.xm , exp.(meanfun.omega)
  z2 = @. ((X-xm)/omega )^2
  if !isnothing(grad)
    midgrad = @.  (xm-X)/(omega^2)
    grad .= vcat(1.0, midgrad , -z2)  # no clue why the gradient is this :-(
  end
  @. x0 + 0.5*z2  # broadcast magic!
end
function _get_mfun_grad(X::Vector{Float64},meanfun::GPNegQuad,
                  grad::T) where T<:Union{Nothing,AbstractVector}
  _test_Xgrad_size(grad,X,meanfun)
  x0, xm , omega = meanfun.x0 , meanfun.xm , exp.(meanfun.omega)
  z2 = @. ((X-xm)/omega )^2
  if !isnothing(grad)
    midgrad = @.  (X-xm)/(omega^2)
    grad .= vcat(1.0, midgrad , z2)  # no clue why the gradient is this :-(
  end
  @. x0 - 0.5*z2  # broadcast magic!
end
function _get_mfun_grad(X::Vector{Float64},
                  meanfun::MM, grad::T) where {
                          T<:Union{Nothing,AbstractVector} , MM<:Union{GPSE,GPNegSE}}
  _test_Xgrad_size(grad,X,meanfun)
  x0, hm xm ,  omega = meanfun.x0 , meanfun.h, meanfun.xm , exp.(meanfun.omega)
  z2 = @. ((X-xm)/omega )^2
  if isa(MM,GPSE)
    se = @. h * exp(-0.5*z2)
  else 
    se = @. -h * exp(-0.5*z2) # must be a NegSE
  end
  if !isnothing(grad)
    midgrad = @.  se*(X-xm)/(omega^2)
    bottomgrad = @. z2*se
    grad .= vcat(1.0, se, midgrad , bottomgrad)  # no clue why the gradient is this :-(
  end
  @. x0 + se  # broadcast magic!
end


function booh(x::A  ,y::B  ) where {A<:String , B<:Integer}
  return true
end



"""
        gplite_meanfun(meanfun::T,X,
            (grad::Union{Nothing,Matrix{Float64})=nothing ) where T<:GPLMeanFun

computes the GP mean function evaluated at test points X. If a gradient matrix is
provided as third argument, it is filled by the gradients for each hyperparameter.
The gradient matrix should have size `n` ``\times`` `nhyp`. Where n is the length of X
and `nhyp` is the number of hyperparameters (see function `nhyper`).
"""
function gplite_meanfun(meanfun::T,X::Vector{Float64},
            (grad::Union{Nothing,Matrix{Float64})=nothing ) where T<:GPLMeanFun

  dims = length(X)
  _test_grad_size(grad,meanfun)

end
