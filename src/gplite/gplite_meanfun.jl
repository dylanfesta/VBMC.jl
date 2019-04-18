

abstract type GPLMeanFun end

# The mean function structs include the fit parameters
# The fit function is a different version of the constructor for each

struct GPZero <: GPLMeanFun  end # no parameters! All empty!

struct GPConst <: GPLMeanFun
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
struct GPSE <: GPLMeanFun
  m0::Float64
  xm::Vector{Float64}
  omega::Vector{Float64}
  h::Float64
  x0::Vector{Float64}
  LB::Vector{Float64}
  UB::Vector{Float64}
  PLB::Vector{Float64}
  PUB ::Vector{Float64}
end
struct GPNegSE <: GPLMeanFun
  m0::Float64
  xm::Vector{Float64}
  omega::Vector{Float64}
  h::Float64
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

# auxiliary functions to test the sizes of the gradient vector
function _test_grad_size(grad::Nothing, meanfun::GPLMeanFun)
  return nothing # with no grad the test is always passed
end
function _test_grad_size(grad::AbstractVector,meanfun::GPLMeanFun)
  nh = length(meanfun)
  ng = length(grad)
  @assert nh == ng " number of hyperparamters and grad vector size do not match"
  nothing
end


# now the fitting of the mean funtions

# auxiliary function that
# copies parameters across, resuling in a hierarchical structure
function _copypars!(sourc::GPLMeanFun, x0::V,LB::V,UB::V,
                PLB::V,PUB::V) where V<:AbstractVector{Real}
  nn = minimum(length.([ dest.x0, x0 ]))
  dest.x0[1:nn] = x0[1:nn]
  dest.LB[1:nn] = LB[1:nn]
  dest.UB[1:nn] = UB[1:nn]
  dest.PLB[1:nn] =PLB[1:nn]
  dest.PUB[1:nn] =PUB[1:nn]
end

function GPZero(X::Matrix{Float64},y::Vector{Float64})
  return GPZero()
end
function GPConst(X::Matrix{Float64},y::Vector{Float64})
  l = 1
  ymin,ymax = extrema(y)
  h = ymax-ymin
  LB = Float64[ ymin - 0.5h ] #defined as 1 vectors, not scalars
  UB = Float64[ ymax + 0.5h ]
  PLB,x0,PUB = [Float64[q] for q in  quantile(y,[0.1, 0.5 , 0.9]) ]
  GPConst(h,x0,LB,UB,PLB,PUB)
end
function GPLinear(X::Matrix{Float64},y::Vector{Float64})
  bb=exp(3.0)
  l = 1+size(X,1)
  (x0,LB,UB,PLB,PUB) = [ Vector{Float64}(undef,l) for _ in 1:5]
  _gpaux = GPConst(X,y)
  _copypars!(_gpaux,x0,LB,UB,PLB,PUB)
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
  l = 1+2size(X,1)
  (x0,LB,UB,PLB,PUB) =  [ Vector{Float64}(undef,l) for _ in 1:5]
  _gpaux = GPLinear(X,y)
  _copypars!(_gpaux ,x0,LB,UB,PLB,PUB)
  w = _gpaux.w
  h = _gpaux.h
  delta = w ./ h
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
  (x0,LB,UB,PLB,PUB) =  [ Vector{Float64}(undef,l) for _ in 1:5]

  # second part first
  minx,maxx = let extr =  extrama(X ; dims = 2)[:]
    (first.(extr) , last.(extr))
  end
  w = maxx .- minx
  LB[2:2l+1] = vcat( (@. minx - 0.5w), (@. log(w) + log(tol)) )
  UB[2:2l+1] = vcat( (@. maxx + 0.5w), (@. log(w) + log(bb)) )
  PLB[2:2l+1] = vcat( minx, (@. log(w) + 0.5log(tol)  ) )
  PUB[2:2l+1] = vcat(maxx, log.(w))
  x0[2:2l+1] = vcat( maxx , log(std(X;dims=2))[:])

  # now first parameter
  miny,maxy = extrema(y)
  h = maxy-miny
  LB[1] , UB[1] = miny, maxy + h
  PLB[1] , PUB[1]  = median(y) , maxy
  x0[1] = quantile(y,0.9)
  GPPosQuad(xm, w ,h , LB,UB,PLB,PUB)
end
function GPNegQuad(X::Matrix{Float64},y::Vector{Float64})
  l = 2+2size(X,1)
  (x0,LB,UB,PLB,PUB) =  [ Vector{Float64}(undef,l) for _ in 1:5]
  _gpaux = GPPosQuad(X,y)
  _copypars!(_gpaux ,x0,LB,UB,PLB,PUB)
  w = _gpaux.xm
  # now first parameter
  miny,maxy = extrema(y)
  h = maxy-miny
  LB[1] , UB[1] = miny - h, maxy
  PLB[1] , PUB[1]  = miny , median(y)
  x0[1] = quantile(y,0.1)
  GPPosQuad(xm, w ,h , LB,UB,PLB,PUB)
end
function GPSE(X::Matrix{Float64},y::Vector{Float64})
  bb=exp(3.0)
  tol=1E-6
  l = 2+2size(X,1)
  (x0,LB,UB,PLB,PUB) =  [ Vector{Float64}(undef,l) for _ in 1:5]
  _gpaux = GPPosQuad(X,y)
  h = _gpaux.something
  w = _gpaux.xm
  _copypars!(_gpaux ,x0,LB,UB,PLB,PUB)
  # last parameter
  LB[2l+2] = log(h) + log(tol)
  UB[2l+2] =  log(h) + log(bb)
  PLB[2l+2] = log(h) +0.5log(tol)
  PUB[2l+2] = log(h)
  x0[2l+2] =  log(std(y))

  # first parameter is the same as pos quad
  miny,maxy = extrema(y)
  h = maxy-miny
  GPSE(xm, w ,h , LB,UB,PLB,PUB)
end
function GPNegSE(X::Matrix{Float64},y::Vector{Float64})
  bb=exp(3.0)
  tol=1E-6
  l = 2+2size(X,1)
  (x0,LB,UB,PLB,PUB) =  [ Vector{Float64}(undef,l) for _ in 1:5]
  _gpaux = GPSE(X,y)
  h = _gpaux.something
  w = _gpaux.xm
  _copypars!(_gpaux ,x0,LB,UB,PLB,PUB)
  # first parameter is the same as neg quad
  _gpaux2 = GPNegQuad(X,y)
  LB[1] , UB[1] = _gpaux2.LB[1],  _gpaux2.UB[1]
  PLB[1] , PUB[1]  = _gpaux2.PLB[1],  _gpaux2.PUB[1]
  x0[1] = _gpaux2.x0[1]
  GPNegSE(xm, w ,h , LB,UB,PLB,PUB)
end

# sub-function to compute and get gradient functions, for each type of mean function.
# (basically all the computation for the mean functions is here)
function gplite_get_mfun_grad(X::Vector{Float64},meanfun::GPLMeanFun,
                grad::T) where T<:Union{Nothing,AbstractVector}
  @error " This function has not been defined for $(typeof(meanfun)) ! "
  return nothing
end
function gplite_get_mfun_grad(X::Vector{Float64},meanfun::GPZero,
                          grad::T) where T<:Union{Nothing,AbstractVector}
  n=length(X)
  if !isnothing(grad)
    fill!(grad,NaN)  # should it be 1 anyway?
  end
  zero(X)
end
function gplite_get_mfun_grad(X::Vector{Float64},meanfun::GPConst,
                      grad::T) where T<:Union{Nothing,AbstractVector}
  n=length(X)
  if !isnothing(grad)
    fill!(grad,1.0)
  end
  fill(meanfun.m0,n)
end
function gplite_get_mfun_grad(X::Vector{Float64},meanfun::GPLinear,
                            grad::T) where T<:Union{Nothing,AbstractVector}
  m0, w = meanfun.m0 , meanfun.w
  if !isnothing(grad)
    grad .= vcat(1.0,X)
  end
  @. m0 + w*X
end
function gplite_get_mfun_grad(X::Vector{Float64},meanfun::GPQuad,
                  grad::T) where T<:Union{Nothing,AbstractVector}
  m0, xm , xmsq = meanfun.m0 , meanfun.xm , meanfun.xmsq
  Xsq = X.^2
  if !isnothing(grad)
    grad .= vcat(1.0, X , Xsq)
  end
  @. m0 + xm*X + xmsq*Xsq # broadcast magic!
end
function gplite_get_mfun_grad(X::Vector{Float64},meanfun::MF,grad::T) where {
                    MF<:Union{GPPosQuad,GPNegQuad} , T<:Union{Nothing,AbstractVector} }
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


# this is the main interface

"""
        gplite_meanfun(meanfun::GPLMeanFun, X , grad)

computes the GP mean function evaluated at test points `X`. If `X` is a vector of
size `d`,  `grad` should either be `nothing` or a vector of size `length(meanfun)`
(i.e. the number of hyperparamters) .   In case of `n` points, both `X` and `grad`
should have `n` columns.
"""
function gplite_meanfun(meanfun::GPLMeanFun,X::V,grad::G)
      where {V<:AbstractVector{Real} , G<: Union{Nothing,AbstractVector{Real}}
  _test_grad_size(grad,meanfun)
  gplite_get_mfun_grad(meanfun,X,grad)
end
function gplite_meanfun(meanfun::GPLMeanFun,X::M,grad::G)
      where {M<:AbstractMatrix{Real} , G<: Union{Nothing,AbstractMatrix{Real}} }
  n = size(X,2)
  out= Vector{Float64}(undef,n)
  for i in 1:n
    _x = view(X,:,i)
    _grad = isnothing(grad) ? nothing : view(grad,:,i)
    out[i] = gplite_get_mfun_grad(meanfun,_x,_grad)
  end
  return out
end
