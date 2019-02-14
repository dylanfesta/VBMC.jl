

# function to deal with parameter transformations


# auxiliary structure for a more modular approach
struct SliceSamplerPars
  samples::Matrix{Float64}
  xx
  xx_sum
  xx_sqsum
  xprime
  xl
  xr
  LB
  UB
  LB_out
  UB_out
  widths
  basewidths
  nthin::Integer
  nburn::Integer
  state::SliceSamplerState
  logpdf::Function
  logprior::Function
  dostepout::Bool
  isadaptive::Bool
  # constructor
  function SliceSamplerPars(logpdf::Function ,
    x0::AbstractVector{Float64} , nsampl::Integer ;
    logprior::Union{Function,Nothing} = nothing ,
    thinning::Integer = 1 ,
    burnin::Union{Integer,Nothing} = nothing,
    step_out = false, display=true,
    widths::Union{Nothing,AbstractVector{Float64},Float64} = nothing ,
    boundaries=(-Inf,+Inf),
    dostepout = false ,
    isadaptive = true)
  D = length(x0)
  function tovec(x::Real)::Vector{Float64}
    fill(x,D)
  end
  function tovec(x::Vector{T} where T<:Real)::Vector{Float64}
    @assert length(x) == D  "something wrong in the vector size"
    return x
  end
  xx,xprime,xl,xr = [ copy(x0) for _ in 1:4]
  xx_sum, xx_sqsum = zero(x0), zero(x0)
  LB, UB  = [ tovec(_x) for _x in boundaries ]
  # basewidths used only for adaptation
  basewidths = something(widths , zero(x))

  widths = if !isnothing(widths)
              tovec(widths)
           else
             deltaB =  @. (UB - LB) / 2
             map(deltaB) do d
               isfinite(d) ? d : 10.0
             end
          end
  LB_out = map(x-> isfinite(x) ? x-eps(x) : x , LB)
  UB_out = map(x-> isfinite(x) ? x+eps(x) : x , UB)
  function _logprior(x::Vector{Float64})::Float64
    if isnothing(logprior)
      return 0.0
    else
      logprior(x)
    end
  end
  samples = Matrix{Float64}(undef,nsampl, D)
  nthin = floor(thinning)
  burn =  something(burnin, div(nsampl,3))
  # now check all conditions
  @assert  all(UB .>= LB)
       "All upper bounds UB need to be equal or greater than lower bounds LB."
  @assert all(basewidths .>= 0 .&  isfinite.(basewidths))
           "The vector widths need to be all positive real numbers."
  @assert all(widths .> 0 .&  isfinite.(widths))
          "The vector widths need to be all positive real numbers."
  @assert all(x0 .>= LB) && all(x0 .<= UB)
         "The initial starting point X0 is outside the bounds."
  @assert isfinite(logpdf(x0))
    "The initial starting point x0 needs to evaluate to a real number (not Inf or NaN)."
  @assert thin > 0
        "The thinning factor OPTIONS.Thin needs to be a positive integer."
  @assert burn >= 0
         "The burn-in samples burnin needs to be a non-negative integer."
  if burn == 0 && isadaptive
    @error(" (non critical) Adaptation is ON but burnin option is set to 0!"*
              "For adaptive width set burnin > 0.")
  end
  new(xx,xx_sum,xx_sqsum,xprime,xl,xr,LB,UB,LB_out,UB_out,widths,basewidths,
       nthin, nburn,logpdf,_logprior, dostepout , isadaptive)
  end
end

# auxiliary function
# check results and bounds of the logpdf function
# the logpdf should return a scalar, and take a vector as input
function logpdf_bounds(x::AbstractVector{Float64},p::SliceSamplerPars)::Float64
  logpri = p.logprior(x)
  fval = p.logpdf(x)
  if  any( ( x .< p.LB) .| ( x .> p.UB)  ) || !isfinite(logpri)
     return -Inf
  elseif isnan(fval)
    @warn "Target density function returned NaN. Trying to continue."
    return -Inf
  end
  fval + logpri
end


# auxiliary function
# defines new left and righ boundaries
# updating x_l and x_r internally
# over dimension d. Using x as starting point
# and taking width as reference
function _update_bounds!(d,x_start,p::SliceSamplerPars)
  rnd = rand()
  wdd = width[d]
  l = x_start - rnd*wdd
  r = x_start + (1. - rnd)*wdd
  LBout,UBout = p.LB_out[d], p.UB_out[d]
  if isfinite(LBout) && l < LBout
    delta = LBout - l
    l += delta
    r += delta
  end
  if isfinite(UBout) && r > LBout
    delta = r - LBout
    r -= delta
    l -= delta
  end
  # maybe this is an overkill
  p.x_l[d] = max(l,LBout)
  p.x_r[d] = min(r,UBout)
  nothing
end


# auxiliary function to sample point in the slice
# shrinks the interval and updates the boundaries when outside of the volume
# the x(prime) vector is updated
function _shrink(d,nshrink,current_logPx::Float64,p::SliceSamplerPars)
  nshrink += 1
  x_try = rand()*(p.x_r[d] - p.x_l[d]) + p.x_l[d]
  p.xprime[d] = x_try
  logPx::Float64 = logpdf_bounds(p.xprime,p)
  isgood = new_logPx < current_logPx
  # the sample is outside,  shrink the boundaries
  if !isgood
    if x_try > xref
        x_r[d] .= x_try
    elseif x_try < xref
        x_l[d] .= x_try
    else # error when xl and xr collapse unto each other
      error("Boundaries shrunk to current position and proposal still not acceptable.")
      # Current position: $xx  Log f: (new value) $log_Px)
      # , (target value)  $log_uprime" )
    end
  end
  return (new_logPx,nshrink,isgood)
end

# auxiliary function, updates left and right boundaries in place
function _do_step_out(d,current_logPx::Float64, p::SliceSamplerPars)
  steps = 0
  if p.dostepout
    while logpdf_bounds(x_l,p) > current_logPx
        x_l[d] .-= stepsize
        steps += 1
    end
    while logpdf_bounds(x_r,p) > current_logPx
        x_r[d] .+= stepsize
        steps += 1
    end
  end
  return steps
end

# auxiliary function
# performs width adaptation during burnin
# does nothing for nshrink = 2 or 3
function _adapt_width_burning!(ii,d,nshrink::Integer,p::SliceSamplerPars)
  # do nothing without burning conditions
  if (ii > p.nburn) || (! p.isadaptive)
    return nothing
  end
  delta_bound = p.UB[d] - p.LB[d]
  w_pre = p.widths[d]
  if nshrink > 3
    delta_eps = isfinite(delta_bound) ? eps(delta_bound) : eps()
    p.widths[d] .= max(w_pre/1.1,delta_eps)
  elseif nshrink < 2
    p.widths[d] .=  min(w_pre, delta_bound)
  end
  nothing
end


# Store summary statistics starting half-way into burn-in
function _update_summary!(ii,p::SliceSamplerPars)
  if ii <= p.burn && ii > div(p.nburn,2)
    p.xx_sum .+=  xx
    p.xx_sqsum .+=  xx.^2
  end
  nothing
end

# auxuliary function, performs width adaptation at the end of the burn-in
# procedure
function _adapt_width_burn_end!(ii,p::SliceSamplerPars)
  if (ii != p.nburn) || (! p.isadaptive)
    return nothing
  end
  burnstored = div(p.nburn,2)
  map!(p.widths, zip(p.xx_sqsum,p.xx_sum,
                   p.UB_out,p.LB_out,p.basewidths)) do (xsq,xs,ub,lb,wbase)
    neww = min( 5.0sqrt(xsq/burnstored - (xs/burnstored)^2) , ub-lb)
    # Max between new widths and geometric mean with user-supplied
    # widths (i.e. bias towards keeping larger widths)
    max(neww,sqrt(neww*wbase))
  end
end


function slicesamplebnd(logpdf::Function ,
  x0::AbstractVector{Float64} , nsampl::Integer ;
  logprior::Union{Function,Nothing} = nothing ,
  thinning::Integer = 1 ,
  burning::Union{Integer,Nothing} = nothing,
  step_out = false, display=true,
  widths = nothing ,
  boundaries=(-Inf,Inf),
  dostepout = false ,
  isadaptive = true )

  D = length(x0)
  function tovec(x::Real,n)::Vector{Float64}
   fill(x,n)
  end
  function tovec(x::Vector{T} where T<:Real , n)
    @assert length(x) == n  "something wrong in the vector size"
    return x
  end

  LB, UB  = [ tovec(x,D) for x in boundaries]
  widths = let _w =  something(widths , @. (UB - LB) / 2 )
     tovec(_w,D) end
  widths[isinf.(widths)] .= 10.0
  widths[LB .== UB] .= 1.0   # WIDTHS is irrelevant when LB == UB, set to 1
  LB_out = @. LB-eps(LB) # this is NaN for UB,LB = Inf
  UB_out = @. UB+eps(UB)

  basewidths = widths
  doprior = !isnothing(logprior)

  # calls the logpdf function also checking the bounds and counting the calls
  funccount = 0
  function _logpdf(x::Vector{Float64})::Float64
    funccount +=1
    lpdf::Float64 = logpdf(x)
    if doprior
      _logpdf_check_bound(LB,UB,x,lpdf, logprior(x))
    else
      _logpdf_check_bound(LB,UB,x,lpdf, 0.0 )
    end
  end

  log_Px = _logpdf(x0)
  xx = x0
  samples = zeros(nsampl, D)

  thin = floor(thinning)
  burn =  something(burning, div(nsampl,3))

  # check things
  @assert  all(UB .>= LB)
       "All upper bounds UB need to be equal or greater than lower bounds LB."
  @assert all(widths .> 0 .&  isfinite.(widths))
          "The vector WIDTHS need to be all positive real numbers."
  @assert all(x0 .>= LB) && all(x0 .<= UB)
         "The initial starting point X0 is outside the bounds."
  @assert isfinite(log_Px)
    "The initial starting point X0 needs to evaluate to a real number (not Inf or NaN)."
  @assert thin > 0
        "The thinning factor OPTIONS.Thin needs to be a positive integer."
  @assert burn >= 0
         "The burn-in samples OPTIONS.Burnin needs to be a non-negative integer."

  effN = nsampl + (nsampl-1)*(thin-1) # effective samples
  # always verbose, will chage later
  println(" Iteration     f-count       log p(x)                   Action")
  showinfo(all...) = let
     _format = " {:4d}         {:8f}    {:12.6e}       {:26s}"
     printfmtln(_format , all...)
  end
  xx_sum = zeros(D)
  xx_sqsum = zeros(D)

  # used to sample between 0 and log(f(x))
  exp_distr = Exponential(1.0)
  # sampling cycle
  for ii in 1:(effN+burn)
    # plot info
    if ii == burn+1
      action = "start recording";
      showinfo(ii-burn,funccount,log_Px,action);
    end
    # Slice-sampling step
    # Random-permutation axes sweep
    for dd in randperm(D)
      # Fixed dimension, skip
      (LB[dd] == UB[dd]) && continue
      # else
      widthdd = widths[dd]
      log_uprime = log_Px - rand(exp_distr)
      x_l, x_r, xprime = ( copy(xx) for _ in 1:3)
      # adjust interval to outside bounds for bounded problems
      x_l[dd] , x_r[dd] = _adjust_bounds(widthdd, LB[dd], UB[dd] ,xx[dd])

      # Step-out procedurem updates x_l and x_r in place
      if dostepout
        steps = _do_step_out(log_uprime,_logpdf,widthdd,x_l,x_r)
        if steps >= 10
          action = "step-out dim $dd  ($steps steps)"
          showinfo(ii-burn,funccount,log_Px,action)
        end
      end
      # Shrink procedure (inner loop)
      # Propose xprime and shrink interval until good one found
      # also, updates log_Px to current value
      (log_Px,shrink,x_l[dd],x_r[dd]) = _do_shrink(0,dd,log_uprime,_logpdf,
                                                      xprime, xx[dd] ,x_l[dd],x_r[dd])
      if shrink >= 10
        action = "shrink dim $dd ($shrink steps)"
        showinfo(displayFormat,ii-burn,funccount,log_Px,action)
      end
      # Width adaptation (only during burn-in, might break detailed balance)
      if ii <= burn && isadaptive
        delta = UB[dd] - LB[dd]
        widths[dd] = _adapt_width(delta,shrink,widthdd)
      end
      # all done , update the dimension
      xx[dd] = xprime[dd]
    end
    # all dimensions completed!
    # record samples and miscellaneous bookkeeping
    do_record = (ii > burn) && (mod(ii - burn - 1, thin) == 0)
    if do_record
      ismpl = 1 + div(ii-burn-1,thin)
      samples[ismpl,:] = xx
        # if nargout > 1; fvals(ismpl,:) = fval(:); end
        # if nargout > 3 && doprior; logpriors(ismpl) = logprior; end
    end
    # Store summary statistics starting half-way into burn-in
    if ii <= burn && ii > div(burn,2)
      xx_sum .+=  xx
      xx_sqsum .+=  xx.^2
      # End of burn-in, update WIDTHS if using adaptive method
      if ii == burn && isadaptive
        burnstored = div(burn,2)
        newwidths = @. 5*sqrt(xx_sqsum/burnstored - (xx_sum/burnstored)^2)
        broadcast!( (n,u,l) -> min(n, u-l),newwidths,  newwidths,UB_out,LB_out)
        if !all(isreal.(newwidths))
           newwidths .= widths
        end
        if isnothing(basewidths)
            widths = newwidths
        else
          # Max between new widths and geometric mean with user-supplied
          # widths (i.e. bias towards keeping larger widths)
          widths =   broadcast( (n,b) -> max(n, sqrt(n*b)) , newwidths, basewidths)
        end
      end
    end
    action = if ii <= burn; "burn"
      elseif !do_record ; "thin"
      else "record"; end
    showinfo(ii-burn,funccount,log_Px,action)
    thinmsg = thin > 1 ? "\n keeping 1 sample every $thin\n"  : "\n"
  end
  println("\nSampling terminated:",
    "$nsampl samples obtained after a burn-in period of $burn samples",
    thinmsg,
    "for a total of $funccount function evaluations")
  return samples
end
