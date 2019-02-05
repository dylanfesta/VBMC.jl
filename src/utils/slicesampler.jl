


function _logpdf_check_bound( LB::Vector{Float64}, UB::Vector{Float64},
        fval::Vector{Float64}, val_logprior::Float64)

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
  l,r = (x_start, x_start)
  rnd = rand()
  l -= rnd*width
  r += (1. - rnd)*width
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
  max(l,LBout) , min(r,LBout)
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
  tovec(x::Vector{T} where T<:Real , n)  = x

  LB,UB =  boundaries
  LB, UB  = [ tovec(x,D) for x in (LB, UB)]
  widths = let _w =  something(widths , @. (UB - LB) / 2 )
     tovec(_w,D) end
  widths[isinf.(widths)] .= 10.0
  widths[LB .== UB] .= 1.0   # WIDTHS is irrelevant when LB == UB, set to 1
  LB_out = @. LB-eps(LB)
  UB_out = @. UB+eps(UB)

  basewidths = widths
  doprior = !isnothing(logprior)
  if !doprior
    logprior(x) = Float64(0.0)
  end

  _logpdf(x) = _logpdf_check_bound(LB,UB,logpdf(x) , logprior(x))

  funccount = 0
  log_Px = _logpdf(x0)
  xx = x0
  samples = zeros(nsampl, D)

  thin = floor(thinning)
  burn =  something(burning, div(nsampl,3))
  @show burn

  # check things
  @assert  all(UB .>= LB)
       "All upper bounds UB need to be equal or greater than lower bounds LB."
  @assert all(widths .> 0 .&  isfinite.(widths))
          "The vector WIDTHS need to be all positive real numbers."
  @assert all(x0 .>= LB) && all(x0 .<= UB)
         "The initial starting point X0 is outside the bounds."
  @assert all(isfinite.(y))
    "The initial starting point X0 needs to evaluate to a real number (not Inf or NaN)."
  @assert thin > 0
        "The thinning factor OPTIONS.Thin needs to be a positive integer."
  @assert burn >= 0
         "The burn-in samples OPTIONS.Burnin needs to be a non-negative integer."

  effN = nsampl + (nsampl-1)*(thin-1) # effective samples
  # for now always verbose
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
        if trace > 1 && steps >= 10
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
      ##  Record samples and miscellaneous bookkeeping
      # Record samples?
      record = ii > burn && mod(ii - burn - 1, thin) == 0
      if record
          ismpl = 1 + (ii - burn - 1)/thin
          samples[ismpl,:] = xx
          # if nargout > 1; fvals(ismpl,:) = fval(:); end
          # if nargout > 3 && doprior; logpriors(ismpl) = logprior; end
      end
      # Store summary statistics starting half-way into burn-in
      if ii <= burn && ii > burn/2
        xx_sum +=  xx
        xx_sqsum +=  xx.^2

        # End of burn-in, update WIDTHS if using adaptive method
        if ii == burn && isadaptive
          burnstored = div(burn,2)
          newwidths = @. 5*sqrt(xx_sqsum/burnstored - (xx_sum/burnstored)^2)
          broadcast!( (n,u,l) -> min(n, u-l),newwidths,  newwidths,UB_out,LB_out)
          if !all(isreal.(newwidths))
             newwidths .= widths end
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
      elseif !record ; "thin"
      else "record"; end
      showinfo(ii-burn,funccount,log_Px,action)
      thinmsg = thin > 1 ? "\n keeping 1 sample every $thin\n"  : "\n"
  end
  println("\nSampling terminated:",
    "$nsampl samples obtained after a burn-in period of $burn samples",
    thinmsg,
    "for a total of $funccount function evaluations"
end
  #
  # % Sanity checks
  # assert(size(LB,1) == 1 && size(UB,1) == 1 && numel(LB) == D && numel(UB) == D, ...
  #     'LB and UB need to be empty matrices, scalars or row vectors of the same size as X0.');
  #=
  if options.Burnin == 0 && isempty(widths) && options.Adaptive
  warning('WIDTHS not specified and adaptation is ON (OPTIONS.Adaptive == 1), but OPTIONS.Burnin is set to 0. SLICESAMPLEBND will attempt to use default values for WIDTHS.');
end
=#

slicesamplebnd(identity , [1.,2.] , 10 ; boundaries = ([1.,2.] , [42. , 78  ]) ,
  widths=4.0, burning=2 )
