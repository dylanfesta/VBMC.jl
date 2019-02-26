

abstract type GPLMeanFun end

# The mean function structs include the fit parameters
# The fit function is a different version of the constructor for each

struct GPZero <: GPLMeanFun
  x0::Vector{Float64}
  LB::Vector{Float64}
  UB::Vector{Float64}
  PLB::Vector{Float64}
  PUB ::Vector{Float64}
 end
struct GPConst <: GPLMeanFun
  c::Float64
  m0::Vector{Float64}
  x0::Vector{Float64}
  LB::Vector{Float64}
  UB::Vector{Float64}
  PLB::Vector{Float64}
  PUB ::Vector{Float64}
end
struct GPLinear <: GPLMeanFun
  m0::Float64
  w::Vector{Float64}
  x0::Vector{Float64}
  LB::Vector{Float64}
  UB::Vector{Float64}
  PLB::Vector{Float64}
  PUB ::Vector{Float64}
end
struct GPQuad <: GPLMeanFun
  m0::Float64
  xm::Vector{Float64}
  xmsq::Vector{Float64}
  x0::Vector{Float64}
  LB::Vector{Float64}
  UB::Vector{Float64}
  PLB::Vector{Float64}
  PUB ::Vector{Float64}
end
struct GPNegQuad <: GPLMeanFun
  m0::Float64
  xm::Vector{Float64}
  omega::Vector{Float64}
  x0::Vector{Float64}
  LB::Vector{Float64}
  UB::Vector{Float64}
  PLB::Vector{Float64}
  PUB ::Vector{Float64}
end
struct GPPosQuad <: GPLMeanFun
  m0::Float64
  xm::Vector{Float64}
  omega::Vector{Float64}
  x0::Vector{Float64}
  LB::Vector{Float64}
  UB::Vector{Float64}
  PLB::Vector{Float64}
  PUB ::Vector{Float64}
end
struct GPSE <: GPLMeanFun
  m0::Float64
  h::Float64
  xm::Vector{Float64}
  omega::Vector{Float64}
  x0::Vector{Float64}
  LB::Vector{Float64}
  UB::Vector{Float64}
  PLB::Vector{Float64}
  PUB ::Vector{Float64}
end
struct GPNegSE <: GPLMeanFun
  m0::Float64
  h::Float64
  xm::Vector{Float64}
  omega::Vector{Float64}
  x0::Vector{Float64}
  LB::Vector{Float64}
  UB::Vector{Float64}
  PLB::Vector{Float64}
  PUB ::Vector{Float64}
end


"""
        length(mfun::GPLMeanFun)
returns the number of hyperparameter of the given GPLite mean function
"""
function Base.length(mfun::GPLMeanFun)
    @error "Base.length not defined for this category of mean function!!!"
    nothing
end
function Base.length(mfun::GPZero)
    return 0
end
function Base.length(mfun::GPConst)
    return 1
end
function Base.length(mfun::GPLinear)
    return 1+length(mfun.w)
end
function Base.length(mfun::G) where G<:Union{GPQuad,GPNegQuad,GPPosQuad,GPSE,GPNegSE}
    return 1+2length(mfun.xm)
end
function Base.length(mfun::G) where G<:Union{GPSE,GPNegSE}
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

# auxuliary compute and get gradient functions, for each type of mean function.
function gplite_get_mfun_grad(X::Vector{Float64},meanfun::GPLMeanFun,
                grad::T) where T<:Union{Nothing,AbstractVector}
  @error " This function has not been defined for $(typeof(meanfun)) ! "
  return nothing
end
function gplite_get_mfun_grad(X::Vector{Float64},meanfun::GPZero,
                          grad::T) where T<:Union{Nothing,AbstractVector}
  n=length(X)
  _test_grad_size(grad,n,meanfun)
  if !isnothing(grad)
    fill!(grad,NaN)  # should it be 1 anyway?
  end
  zero(X)
end
function gplite_get_mfun_grad(X::Vector{Float64},meanfun::GPConst,
                      grad::T) where T<:Union{Nothing,AbstractVector}
  n=length(X)
  _test_grad_size(grad,n,meanfun)
  if !isnothing(grad)
    fill!(grad,1.0)
  end
  fill(meanfun.m0,n)
end
function gplite_get_mfun_grad(X::Vector{Float64},meanfun::GPLinear,
                            grad::T) where T<:Union{Nothing,AbstractVector}
  _test_Xgrad_size(grad,X,meanfun)
  m0, w = meanfun.m0 , meanfun.w
  if !isnothing(grad)
    grad .= vcat(1.0,X)
  end
  @. m0 + w*X
end
function gplite_get_mfun_grad(X::Vector{Float64},meanfun::GPQuad,
                  grad::T) where T<:Union{Nothing,AbstractVector}
  _test_Xgrad_size(grad,X,meanfun)
  m0, xm , xmsq = meanfun.m0 , meanfun.xm , meanfun.xmsq
  Xsq = X.^2
  if !isnothing(grad)
    grad .= vcat(1.0, X , Xsq)
  end
  @. m0 + xm*X + xmsq*Xsq # broadcast magic!
end
function gplite_get_mfun_grad(X::Vector{Float64},meanfun::MF,grad::T) where {
                    MF<:Union{GPPosQuad,GPNegQuad} , T<:Union{Nothing,AbstractVector} }
  _test_Xgrad_size(grad,X,meanfun)
  m0, xm , omega = meanfun.m0 , meanfun.xm , exp.(meanfun.omega)
  # sign for gradient (inverted)
  signfact = MF == GPPosQuad ? (-1.0) : (+1.0)
  z2 = @. ((X-xm)/omega )^2
  if !isnothing(grad)
    midgrad = @.  (X-Xm)/(omega^2)
    # first element of grad always +1
    grad .= signfact .*vcat(signfact, midgrad , z2)
  end
  m0 - signfact*0.5*sum(z2) # this is a scalar !?
end
function gplite_get_mfun_grad(X::Vector{Float64},
                  meanfun::MM, grad::T) where {
                          T<:Union{Nothing,AbstractVector} , MF<:Union{GPSE,GPNegSE}}
  _test_Xgrad_size(grad,X,meanfun)
  m0, h, xm ,  omega = meanfun.m0 , meanfun.h, meanfun.xm , exp.(meanfun.omega)
  z2 = @. ((X-xm)/omega )^2
  signfact = MF == GPSE ? (+1.0) : (-1.0)
  se = mapreduce(_z-> signfact*h*exp(-0.5*_z), + , z2)
  if !isnothing(grad)
    midgrad = @.  se*(X-xm)/(omega^2)
    bottomgrad = @. z2*se
    grad .= vcat(1.0, se, midgrad , bottomgrad)
  end
  m0 + se # yet again, scalar result
end


# now the fitting of the mean funtions

function _gplite_fitbase(n)
  filln(x) = fill(x,n)
  LB, UB = filln(-Inf) , filln(Inf)
  PLB, PUB =  filln(-Inf) , filln(Inf)
  x0 = filln(0.0)
  return (x0,LB,UB,PLB,PUB)
end
function _gplite_fitconst(X,y)
  ymin,ymax = extrema(y)
  h = ymax-ymin
  LB1 = ymin - 0.5h
  UB1 = ymax + 0.5h
  PLB1,PUB1 = quantile(y,[1.0 , 0.9])
  x01 = median.(y)
  return (h,x01,LB1,UB1,PLB1,PUB1)
end

function _gplite_fitomega(X,bign=exp(3.0),tol=1E-6)
  minx = minimum(X ; dims = 2)[:]
  maxx = minimum(X ; dims = 2)[:]
  w = maxx .- minx
  LB = vcat( (@. minx - 0.5w), (@. log(w) + log(tol)) )
  UB = vcat( (@. maxx + 0.5w), (@. log(w) + log(bign)) )
  PLB = vcat( minx, (@. log(w) + 0.5log(tol)  ) )
  PUB = vcat(maxx, log.(w))
  x0 = vcat( maxx , log(std(X;dims=2))[:])
  (x0,LB,UB,PLB,PUB)
end

function GPZero(X::Matrix{Float64},y::Vector{Float64})
  l = 0
  GPZero(_gplite_fitbase(l)...)
end
function GPConst(X::Matrix{Float64},y::Vector{Float64})
  l = 1
  (x0,LB,UB,PLB,PUB) = _gplite_fitbase(l)
  (h,x0[1],LB[1],UB[1],PLB[1],PUB[1]) = _gplite_fitconst(l)
  GPConst(h,x0,LB,UB,PLB,PUB)
end
function GPLinear(X::Matrix{Float64},y::Vector{Float64})
  bb=exp(3.0)
  l = 1+size(X,1)
  (x0,LB,UB,PLB,PUB) = _gplite_fitbase(l)
  (h,x0[1],LB[1],UB[1],PLB[1],PUB[1]) = _gplite_fitconst(l)
  w = [ xu-xl for (xl,xu) in extrema(X;dims=2) ] # rows are dimensions columns are points
  delta = w ./ h
  @. LB[2:l+1] =  -delta *bb
  @. UB[2:l+1] = delta * bb
  @. PLB[2:l+1] =  -delta
  @. PUB[2:l+1] = delta
  GPLinear(h,w,x0,LB,UB,PLB,PUB)
end
function GPQuad(X::Matrix{Float64},y::Vector{Float64})
  bb=exp(3.0)
  tol=1E-6
  l = 1+2size(X,1)
  (x0,LB,UB,PLB,PUB) = _gplite_fitbase(l)
  (h,x0[1],LB[1],UB[1],PLB[1],PUB[1]) = _gplite_fitconst(l)
  w = [ xu-xl for (xl,xu) in extrema(X;dims=2) ]
  delta = w ./ h
  @. LB[2:l+1] =  -delta *bb
  @. UB[2:l+1] = delta * bb
  @. PLB[2:l+1] =  -delta
  @. PUB[2:l+1] = delta
  @. LB[l+2:end] =  -(delta *bb)^2
  @. UB[l+2:end] = (delta * bb)^2
  @. PLB[l+2:end] =  -delta^2
  @. PUB[l+2:end] = delta^2
  GPQuad(h,w,x0,LB,UB,PLB,PUB)
end
function GPPosQuad(X::Matrix{Float64},y::Vector{Float64})
  bb=exp(3.0)
  tol=1E-6
  l = 2+2size(X,1)
  (x0,LB,UB,PLB,PUB) = _gplite_fitbase(l)
  miny,maxy = extrema(y)
  h = maxy-miny
  LB[1] , UB[1] = miny, maxy + h
  PLB[1] , PUB[1]  = median(y) , maxy
  x0[1] = quantile(y,0.9)
  # xm and omega
  for (eq1,eq2) in zip( (x0[3:end], LB[3:end], UB[3:end] , PLB[3:end] , PUB[3:end]) ,
        _gplite_fitomega(X,bb,tol) )
    eq1 .= eq2
  end
end





function gplite_fit_meanfun(X::Matrix{Float64},y::Vector{Float64}, MFType::DataType)
  @assert MFType <: GPLMeanFun
  gplite_fit_meanfun(X,y,Val(MFType))
end

function gplite_fit_meanfun(X,y,::Val{GPZero})

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
