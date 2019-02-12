
# auxiliary function
# check results and bounds of the logpdf function
# the logpdf should return a scalar, and take a vector as input
function _logpdf_check_bound( LB::Vector{Float64},
  UB::Vector{Float64}, x::Vector{Float64},
  fval::Float64, val_logprior::Float64)
  #checking bounds here
  if  any( (x.< LB) .| (x.> UB)  ) || !isfinite(val_logprior)
     return -Inf
  elseif isnan(fval)
    @warn "Target density function returned NaN. Trying to continue."
    return -Inf
  end
  fval + val_logprior
end

# auxiliary function
# defines new left and righ boundaries
# over one dimension. Using x0 and width
function _adjust_bounds(width,LBout,UBout,x0)
  rnd = rand()
  l = x0 - rnd*width
  r = x0 + (1. - rnd)*width
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
  max(l,LBout) , min(r,UBout)
end

# auxiliary function to sample point in the slice
# shrinks the interval and updates the boundaries when outside of the volume
# the x(prime) vector is updated
function _do_shrink(nshrink::Integer,
    idx::Integer,
    targ_lpdf::Float64,
    lpdf::Function, x::AbstractVector{Float64}, xref::Float64,
    xl::Float64,xr::Float64)
  nshrink += 1
  x[idx] = rand()*(xr - xl) + xl
  # the sample is inside , return
  logPx::Float64 = lpdf(x)
  if  logPx < targ_lpdf
    return (logPx,nshrink,xl,xr)
  end
  # the sample is outside,  shrink the boundaries, repeat
  if x[idx] > xref
      xr = x[idx]
  elseif x[idx] < xref
      xl = x[idx]
  else # error when xl and xr collapse unto each other
      error("Shrunk to current position and proposal still not acceptable.")
      # Current position: $xx  Log f: (new value) $log_Px)
      # , (target value)  $log_uprime" )
  end
  _do_shrink(nshrink,idx,targ_lpdf,lpdf,x, xref, xl,xr)
end

# auxiliary function, updates left and right boundaries in place
function _do_step_out(targ_lpdf::Float64,lpdf::Function, stepsize::Float64,
      xl::Vector{Float64},xr::Vector{Float64})
  steps = 0
  while lpdf(x_l) > targ_lpdf
      x_l[dd] .-= stepsize
      steps += 1
  end
  while lpdf(x_r) > targ_lpdf
      x_r[dd] .+= stepsize
      steps += 1
  end
  return steps
end

# auxiliary function , with adaptation and during burn-in
# proposes a new width given the old one
function _adapt_width(delta_bound::Float64,
        nshrink::Integer,oldwidth::Float64)
  if nshrink > 3
    delta_eps = isfinite(delta_bound) ? eps(delta_bound) : eps()
    return max(oldwidth/1.1,delta_eps)
  elseif nshrink < 2
    return min(oldwidth, delta_bound)
  else
    return oldwidth
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
