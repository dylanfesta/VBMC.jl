

using Formatting

function _logpdf_check_bound( LB::Vector{Float64}, UB::Vector{Float64},
        fval::Vector{Float64}, val_logprior::Float64)
  #checking bounds here
  ( any( x.< LB .| x.> UB ) || !isfinite(val_logprior) )  && return -Inf
  if any(isnan.(fval))
    @warn "Target density function returned NaN. Trying to continue."
    return -Inf
  end
  sum(fval) + val_logprior
end

# takes left and righ step based on width , then makes sure
# they are withn the bounds
function _adjust_bounds(width,LBout,UBout,x_start)
  rnd = rand()
  l = x_start -  rnd*width
  r = x_start + (1. - rnd)*width
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
  max(l,LBout) , min(r,RBout)
end

function slicesamplebnd(logpdf::Function , x0::XT , nsampl::Integer ;
  logprior::Union{Function,Nothing} = nothing ,
  thinning::Integer = 1 ,
  burning::Union{Integer,Nothing} = nothing,
  step_out = false, display=true,
  widths = nothing ,
  boundaries=(-Inf,Inf),
  dostepout = false ,
  isadaptive = true ) where XT<:Union{AbstractArray{<:Real},Real}

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

  funccount = 0
  function _logpdf(x)
    funccount +=1
    if doprior
      _logpdf_check_bound(LB,UB,logpdf(x) , logprior(x))
    else
      _logpdf_check_bound(LB,UB,logpdf(x) , 0.0 )
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
  @assert all(isfinite.(log_Px))
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
      log_uprime = log(rand()) + log_Px
      x_l, x_r, xprime = ( copy(xx) for _ in 1:3)
      # adjust interval to outside bounds for bounded problems
      x_l[dd] , x_r[dd] = _adjust_bounds(widthdd, LB[dd], UB[dd] ,xx[dd])

      # Step-out procedure
      if dostepout
        steps = 0
        stepsize = widthdd
        while _logpdf(x_l) > log_uprime
            x_l[dd] .-= stepsize
            steps += 1
        end
        while _logpdf(x_r) > log_uprime
            x_r[dd] .+= stepsize
            steps += 1
        end
        if steps >= 10
            action = "step-out dim $dd  ($steps steps)"
            showinfo(ii-burn,funccount,log_Px,action)
        end
      end
      # Shrink procedure (inner loop)
      # Propose xprimes and shrink interval until good one found
      shrink = 0
      while 1
        shrink += 1
        xprime[dd] = rand()*(x_r[dd] - x_l[dd]) + x_l[dd]
        log_Px = _logpdf(xprime)
        log_Px > log_uprime && break # this is the only way to leave the while loop
        # Shrink in
        if xprime[dd] > xx[dd]
            x_r[dd] = xprime[dd]
        elseif xprime[dd] < xx[dd]
            x_l[dd] = xprime[dd]
        else
            error("Shrunk to current position and proposal still not acceptable.
                Current position: $xx  Log f: (new value) $log_Px) , (target value)  $log_uprime" )
        end
      end
      # Width adaptation (only during burn-in, might break detailed balance)
      if ii <= burn && isadaptive
        delta = UB[dd] - LB[dd];
        if shrink > 3
          if isfinite(delta)
            widths[dd] = max(widths[dd]/1.1,eps(delta))
          else
            widths[dd] = max(widths[dd]/1.1,eps())
          end
        elseif shrink < 2
          widths[dd] = min(widths[dd]*1.2, delta)
        end
      end
      if shrink >= 10
        action = "shrink dim $dd ($shrink steps)"
        showinfo(displayFormat,ii-burn,funccount,log_Px,action)
      end

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
      xx_sum +=  xx
      xx_sqsum +=  xx.^2

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
          widths =   broadcast( (n,b) -> max(n, sqrt(n*b)) ,
                                    newwidths,basewidths)
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
    "for a total of $funccount function evaluations"
end
