

# struct to deal with changing parameters
mutable struct SliceSamplerState
    current_logPx::Float64
    nfunccount::Int64
    nshrink::Int64
end
function SliceSamplerState()
  SliceSamplerState(0.0,0,0)
end

# auxiliary structure for a more modular approach
struct SliceSamplerPars
  samples::Matrix{Float64}
  xx::Vector{Float64}
  xx_sum::Vector{Float64}
  xx_sqsum::Vector{Float64}
  xprime::Vector{Float64}
  x_l::Vector{Float64}
  x_r::Vector{Float64}
  LB::Vector{Float64}
  UB::Vector{Float64}
  LB_out::Vector{Float64}
  UB_out::Vector{Float64}
  widths::Vector{Float64}
  basewidths::Vector{Float64}
  nthin::Integer
  nburn::Integer
  logpdf::Function
  logprior::Function
  dostepout::Bool
  isadaptive::Bool
  state::SliceSamplerState
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
  basewidths = something(widths , zero(xx))

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
  samples = Matrix{Float64}(undef,D,nsampl)
  nthin = floor(thinning)
  nburn =  something(burnin, div(nsampl,3))
  # compute logpdf once
  state = SliceSamplerState()
  state.current_logPx = logpdf(x0)
  # now check all conditions
  @assert  all(UB .>= LB)
       "All upper bounds UB need to be equal or greater than lower bounds LB."
  @assert all(basewidths .>= 0 .&  isfinite.(basewidths))
           "The vector widths need to be all positive real numbers."
  @assert all(widths .> 0 .&  isfinite.(widths))
          "The vector widths need to be all positive real numbers."
  @assert all(x0 .>= LB) && all(x0 .<= UB)
         "The initial starting point X0 is outside the bounds."
  @assert isfinite(state.current_logPx)
    "The initial starting point x0 needs to evaluate to a real number (not Inf or NaN)."
  @assert nthin > 0
        "The thinning factor OPTIONS.Thin needs to be a positive integer."
  @assert nburn >= 0
         "The burn-in samples burnin needs to be a non-negative integer."
  if nburn == 0 && isadaptive
    @error(" (non critical) Adaptation is ON but burnin option is set to 0!"*
              "For adaptive width set burnin > 0.")
  end
  new(samples, xx,xx_sum,xx_sqsum,xprime,xl,xr,LB,UB,LB_out,UB_out,widths,basewidths,
       nthin, nburn,logpdf,_logprior, dostepout , isadaptive,state)
  end
end

# auxiliary function
# check results and bounds of the logpdf function
# the logpdf should return a scalar, and take a vector as input
function logpdf_bounds(x::AbstractVector{Float64},p::SliceSamplerPars)::Float64
  logpri = p.logprior(x)
  fval = p.logpdf(x)
  p.state.nfunccount += 1
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
# over dimension d. Using p.xx as starting point
# and taking width as reference
function _update_bounds!(d::Integer,p::SliceSamplerPars)
  x_start = p.xx[d]
  rnd = rand()
  wdd = p.widths[d]
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
function _shrink!(d,slice_logPx::Float64,p::SliceSamplerPars)
  p.state.nshrink += 1
  x_try = rand()*(p.x_r[d] - p.x_l[d]) + p.x_l[d]
  xref = p.xx[d]
  p.xprime[d] = x_try
  new_logPx::Float64 = logpdf_bounds(p.xprime,p)
  isgood = new_logPx < slice_logPx
  # the sample is outside,  shrink the boundaries
  if !isgood
    if x_try > xref
        p.x_r[d] = x_try
    elseif x_try < xref
        p.x_l[d] = x_try
    else # error when xl and xr collapse unto each other
      error("Boundaries shrunk to current position and proposal still not acceptable.")
      # Current position: $xx  Log f: (new value) $log_Px)
      # , (target value)  $log_uprime" )
    end
  end
  return (new_logPx,isgood)
end

# auxiliary function, updates left and right boundaries in place
function _do_step_out!(d, slice_logPx::Float64 , p::SliceSamplerPars)
  steps = 0
  stepsize = p.widths[d]
  if p.dostepout
    while logpdf_bounds(x_l,p) > slice_logPx
        x_l[d] .-= stepsize
        steps += 1
    end
    while logpdf_bounds(x_r,p) > slice_logPx
        x_r[d] .+= stepsize
        steps += 1
    end
  end
  return steps
end

# auxiliary function
# performs width adaptation during burnin
# does nothing for nshrink = 2 or 3
function _adapt_width_burning!(ii,d,p::SliceSamplerPars)
  # do nothing without burning conditions
  if (ii > p.nburn) || (! p.isadaptive)
    return nothing
  end
  delta_bound = p.UB[d] - p.LB[d]
  w_pre = p.widths[d]
  nshrink = p.state.nshrink
  if nshrink > 3
    delta_eps = isfinite(delta_bound) ? eps(delta_bound) : eps()
    p.widths[d] = max(w_pre/1.1,delta_eps)
  elseif nshrink < 2
    p.widths[d] =  min(w_pre, delta_bound)
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

# for printing on screen
function _showinfo_header()
  println(" Iteration     f-count       log p(x)                   Action")
end
function _showinfo(all...)
   _format = " {:4d}         {:8f}    {:12.6e}       {:26s}"
   printfmtln(_format , all...)
end



function slicesamplebnd(args... ; namedargs... )
  # initialize all variables
  pp = SliceSamplerPars(args... ; namedargs...)
  D,nsampl = size(pp.samples)
  # effective number of sampling steps
  effN = nsampl + (nsampl-1)*(pp.nthin-1)

  # used to sample between 0 and log(f(x))
  exp_distr = Exponential(1.0)

  _showinfo_header()
  # sampling cycle
  for ii in 1:(effN+pp.nburn)
    # plot info
    if ii == pp.nburn+1
      action = "start recording"
      _showinfo(ii-pp.burn,pp.state.nfunccount,
                    pp.state.current_logPx ,action);
    end
    # Slice-sampling step
    # Random-permutation axes sweep
    for dd in randperm(D)
      # Fixed dimension, skip
      (pp.LB[dd] == pp.UB[dd]) && continue
      # else
      # defines new left and right bounds around previous sample xx
      _update_bounds!(dd,pp)
      # picks the "height" of the new slice
      log_Pxslice = pp.state.current_logPx - rand(exp_distr)
      # Step-out procedure (if included), updates x_l and x_r in place
      steps = _do_step_out!(dd,log_Pxslice,pp)
      if steps >= 10
        action = "step-out dim $dd  ($steps steps)"
        showinfo(ii-pp.nburn,pp.state.nfunccount,pp.state.current_logPx, action)
      end
      # shrink procedure,  that also updates xprime
      shrink_done = false
      pp.state.nshrink = 0
      while !shrink_done
        (new_logPx,shrink_done)=_shrink!(dd,log_Pxslice,pp)
      end
      if pp.state.nshrink >= 10
        action = "shrink dim $dd ($shrink steps)"
        showinfo(ii-pp.nburn,pp.state.funccount,pp.state.current_logPx, action)
      end
      # width adapatation during burn in
      _adapt_width_burning!(ii,dd,pp)
      # all done, update the dimension
      pp.xx[dd] = pp.xprime[dd]
       # and reset the boundaries too
      for x in (pp.x_l, pp.x_r)
        copy!(x,pp.xx)
      end
      # TODO  test that I will remvoe later
      @assert all(pp.xx .== pp.xprime)
      pp.state.current_logPx = logpdf_bounds(pp.xx,pp)
    end #all dimensions done!
  end # all ssamples done!
  # exit message and return value
  println("\nSampling terminated:",
    "$nsampl samples obtained after a burn-in period of $(pp.nburn) samples",
    # thinmsg,
    "for a total of $(pp.state.nfunccount) function evaluations")
  return pp.samples
end



#=
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
=#
