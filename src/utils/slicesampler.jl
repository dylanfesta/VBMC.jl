


function eval_logpdf()
  if any( x.< LB .| x.> UB )
    return -Inf
  end
  val_logprior = logprior(x)
  if  ~isfinite(val_logprior)
    return -Inf
  end
  fval = logpdf(x)
  if any(isnan.(fval))
    @warn "Target density function returned NaN. Trying to continue."
    return -Inf
  end
  sum(fval) + val_logprior
end


function slicesamplebnd(logpdf::Function , x0::XT , nsampl::Integer ;
  logprior = nothing ,
  thinning::Integer = 1 ,
  burning::Union{Integer,Nothing} = nothing,
  step_out = false, display=true,
  widths = nothing ,
  boundaries=(-Inf,Inf) ) where XT<:Union{AbstractArray{<:Real},Real}

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

  funccount = 0
  _fake_eval_logpdf(_) = 666.0
  log_Px = _fake_eval_logpdf(x0)
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
      LB[dd] == UB[dd] && continue
      widthdd = widths[dd]

      log_uprime = log(rand()) + log_Px
      x_l, x_r, xprime = ( copy(xx) for _ in 1:3)

      # Create a horizontal interval (x_l, x_r) enclosing xx
      rr = rand()
      x_l[dd] .-= rr*widthdd
      x_r[dd] .+= (1.0 - rr) * widthdd

      # adjust interval to outside bounds for bounded problems
      if isfinite(LB[dd]) || isfinite(UB[dd])
        if x_l[dd] < LB_out[dd]
          delta = LB_out[dd] - x_l[dd]
          x_l[dd] .= x_l[dd] + delta
          x_r[dd] .= x_r[dd] + delta
        end
        if  x_r[dd] > UB_out[dd]
          Vdelta = x_r[dd] - UB_out[dd]
          x_l[dd] = x_l[dd] - delta
          x_r[dd] = x_r[dd] - delta
        end
        # x_l[dd] .= max(x_l[dd],LB_out[dd]) # they seem redundant to me
        # x_r[dd] .= min(x_r[dd],UB_out[dd])
      end
      # Step-out procedure
      if dostepout
        steps = 0
        stepsize = widthdd
        while logpdf(x_l) > log_uprime
            x_l[dd] .-= stepsize
            steps += 1
        end
        while logpdf(x_r) > log_uprime
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
        log_Px = logpdf(xprime)
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
        if ii == burn && insadaptive
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



#=
%% Default parameters and options

% By default unbounded sampling
if nargin < 4; widths = []; end
if nargin < 5 || isempty(LB); LB = -Inf; end
if nargin < 6 || isempty(UB); UB = Inf; end
if nargin < 7; options = []; end

% Default options
defopts.Thin        = 1;            % Thinning
defopts.Burnin      = round(N/3);   % Burn-in
defopts.StepOut     = false;        % Step-out
defopts.Display     = 'notify';     % Display
defopts.Adaptive    = true;         % Adaptive widths
defopts.LogPrior    = [];           % Log prior over X
defopts.Diagnostics = true;         % Convergence diagnostics at the end

for f = fields(defopts)'
    if ~isfield(options,f{:}) || isempty(options.(f{:}))
        options.(f{:}) = defopts.(f{:});
    end
end


%% Startup and initial checks
D = numel(x0);
if numel(LB) == 1; LB = repmat(LB, [1,D]); end
if numel(UB) == 1; UB = repmat(UB, [1,D]); end
if size(LB,1) > 1; LB = LB'; end
if size(UB,1) > 1; UB = UB'; end
if numel(widths) == 1; widths = repmat(widths, [1,D]); end
LB_out = LB - eps(LB);
UB_out = UB + eps(UB);
basewidths = widths;    % User-supplied widths
if isempty(options.LogPrior); doprior = 0; else doprior = 1; end

if options.Burnin == 0 && isempty(widths) && options.Adaptive
    warning('WIDTHS not specified and adaptation is ON (OPTIONS.Adaptive == 1), but OPTIONS.Burnin is set to 0. SLICESAMPLEBND will attempt to use default values for WIDTHS.');
end

if isempty(widths); widths = (UB - LB)/2; end
widths(isinf(widths)) = 10;

funccount = 0;
[y,fval,logprior] = feval(@logpdfbound, x0, varargin{:});   % First evaluation at X0
xx = x0;
samples = zeros(N, D);
if ~isempty(options.LogPrior) && nargout > 3; logpriors = zeros(N, 1); else logpriors = []; end
if nargout > 1; fvals = zeros(N, numel(fval)); end
thin = floor(options.Thin);
burn = floor(options.Burnin);
log_Px = y;
widths(LB == UB) = 1;   % WIDTHS is irrelevant when LB == UB, set to 1

% Sanity checks
assert(size(x0,1) == 1 && size(x0,1) == 1, ...
    'The initial point X0 needs to be a scalar or row vector.');
assert(size(LB,1) == 1 && size(UB,1) == 1 && numel(LB) == D && numel(UB) == D, ...
    'LB and UB need to be empty matrices, scalars or row vectors of the same size as X0.');
assert(all(UB >= LB), ...
    'All upper bounds UB need to be equal or greater than lower bounds LB.');
assert(all(widths > 0 & isfinite(widths)) && isreal(widths), ...
    'The vector WIDTHS need to be all positive real numbers.');
assert(all(x0 >= LB) & all(x0 <= UB), ...
    'The initial starting point X0 is outside the bounds.');
assert(all(isfinite(y)) && isreal(y), ...
    'The initial starting point X0 needs to evaluate to a real number (not Inf or NaN).');
assert(thin > 0 && isscalar(thin), ...
    'The thinning factor OPTIONS.Thin needs to be a positive integer.');
assert(burn >= 0 && isscalar(burn), ...
    'The burn-in samples OPTIONS.Burnin needs to be a non-negative integer.');

effN = N + (N-1)*(thin-1); % Effective samples

switch options.Display
    case {'notify','notify-detailed'}
        trace = 2;
    case {'none', 'off'}
        trace = 0;
    case {'iter','iter-detailed'}
        trace = 3;
    case {'final','final-detailed'}
        trace = 1;
    otherwise
        trace = 1;
end

if trace > 1
    fprintf(' Iteration     f-count       log p(x)                   Action\n');
    displayFormat = ' %7.0f     %8.0f    %12.6g    %26s\n';
end

xx_sum = zeros(1,D);
xx_sqsum = zeros(1,D);



=#
#=
%--------------------------------------------------------------------------
function [y,fval,logprior] = logpdfbound(x,varargin)
%LOGPDFBOUND Evaluate log pdf with bounds and prior.

    fval = [];
    logprior = [];

    if any(x < LB | x > UB)
        y = -Inf;
    else

        if doprior
            logprior = feval(options.LogPrior, x);
            if isnan(logprior)
                y = -Inf;
                warning('Prior density function returned NaN. Trying to continue.');
                return;
            elseif ~isfinite(logprior)
                y = -Inf;
                return;
            end
        else
            logprior = 0;
        end

        fval = logf(x,varargin{:});
        funccount = funccount + 1;

        if any(isnan(fval))
            y = -Inf;
            % warning('Target density function returned NaN. Trying to continue.');
        else
            y = sum(fval) + logprior;
        end
    end
end

=#


#=




% Diagnostics
if options.Diagnostics && (nargout > 2 || trace > 0)
    [exitflag,R,Neff,tau] = diagnose(samples,trace,options);
    diagstr = [];
    if exitflag == -2 || exitflag == -3
        diagstr = [diagstr '\n * Try sampling for longer, by increasing N or OPTIONS.Thin.'];
    elseif exitflag == -1
        diagstr = [diagstr '\n * Try increasing OPTIONS.Thin to obtain more uncorrelated samples.'];
    elseif exitflag == 1
        diagstr = '\n * No violations of convergence have been detected (this does NOT guarantee convergence).';
    end
    if trace > 0 && ~isempty(diagstr)
        fprintf(diagstr);
    end
else
    exitflag = 0;
end

if trace > 0
    fprintf('\n\n');
end

if nargout > 3
    output.widths = widths;
    output.logpriors = logpriors;
    output.funccount = funccount;
    if options.Diagnostics
        output.R = R;
        output.Neff = Neff;
        output.tau = tau;
    end
end


%--------------------------------------------------------------------------
function [y,fval,logprior] = logpdfbound(x,varargin)
%LOGPDFBOUND Evaluate log pdf with bounds and prior.

    fval = [];
    logprior = [];

    if any(x < LB | x > UB)
        y = -Inf;
    else

        if doprior
            logprior = feval(options.LogPrior, x);
            if isnan(logprior)
                y = -Inf;
                warning('Prior density function returned NaN. Trying to continue.');
                return;
            elseif ~isfinite(logprior)
                y = -Inf;
                return;
            end
        else
            logprior = 0;
        end

        fval = logf(x,varargin{:});
        funccount = funccount + 1;

        if any(isnan(fval))
            y = -Inf;
            % warning('Target density function returned NaN. Trying to continue.');
        else
            y = sum(fval) + logprior;
        end
    end
end

end

=#
