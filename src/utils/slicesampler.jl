


function eval_logpdf()
  if any( LB.< x .< UB )
    return -Inf
  end

  val_logprior = logprior(x)
  if  ~isfinite(val_logprior)
    return -Inf
  end

  fval = logpdf(x, varargs...)
  funccount = funccount + 1;

  if any(isnan(fval))
    @warn "Target density function returned NaN. Trying to continue."
    return -Inf
  end
  sum(fval) + val_logprior
end


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
