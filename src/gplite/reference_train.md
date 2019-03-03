
```matlab
function [gp,hyp,output] = gplite_train(hyp0,Ns,X,y,meanfun,hprior,warp,options)
%GPLITE_TRAIN Train lite Gaussian Process hyperparameters.

% Fix functions

if nargin < 5; meanfun = []; end
if nargin < 6; hprior = []; end
if nargin < 7; warp = []; end
if nargin < 8; options = []; end

% Default mean function is constant
if isempty(meanfun); meanfun = 'const'; end

Nopts = [];
if isfield(options,'Nopts'); Nopts = options.Nopts; end
if isempty(Nopts); Nopts = 3; end   % Number of hyperparameter optimization runs

Ninit = [];
if isfield(options,'Ninit'); Ninit = options.Ninit; end
if isempty(Ninit); Ninit = 2^10; end   % Initial design size for hyperparameter optimization

Thin = [];
if isfield(options,'Thin'); Thin = options.Thin; end
if isempty(Thin); Thin = 5; end   % Thinning for hyperparameter sampling

Burnin = [];
if isfield(options,'Burnin'); Burnin = options.Burnin; end
if isempty(Burnin); Burnin = Thin*Ns; end   % Initial design size for hyperparameter optimization

DfBase = [];
if isfield(options,'DfBase'); DfBase = options.DfBase; end
if isempty(DfBase); DfBase = 7; end   % Default degrees of freedom for Student's t prior

Sampler = [];
if isfield(options,'Sampler'); Sampler = options.Sampler; end
if isempty(Sampler); Sampler = 'slicesample'; end   % Default MCMC sampler for hyperparameters

% If using Laplace sampler, always perform optimization
if strcmpi(Sampler,'laplace'); Nopts = max(Nopts,1); end

Widths = [];
if isfield(options,'Widths'); Widths = options.Widths; end
if isempty(Widths); Widths = []; end   % Default widths (used only for HMC sampler)

LogP = [];
if isfield(options,'LogP'); LogP = options.LogP; end
if isempty(LogP); LogP = []; end   % Old log probability associated to starting points


[N,D] = size(X);            % Number of training points and dimension
ToL = 1e-6;

% Set up warped GP
if ~isempty(warp)
    if Ns > 0
        error('gplite_train:NoInputWarpingSampling',...
            'Sampling hyperparameters is currently not supported with input warping.');
    end
    warp.trinfo = warpvars(D,warp.LB,warp.UB);    
    if warp.Nwarp > 0
        warp.trinfo.type = 9*ones(1,D); % Kumaraswamy-logistic transform
        unbnd = (warp.LB == -Inf & warp.UB == Inf);
        warp.trinfo.type(unbnd) = 10;   % Logit-Kumaraswamy-logistic transform
        warp.trinfo.alpha = ones(1,D);
        warp.trinfo.beta = ones(1,D);
    end
    if isfield(warp,'mu') && ~isempty(warp.mu); warp.trinfo.mu = warp.mu; end
    if isfield(warp,'delta') && ~isempty(warp.delta); warp.trinfo.delta = warp.delta; end
    if ~isfield(warp,'logpdf_flag') || isempty(warp.logpdf_flag)
        warp.logpdf_flag = false;
    end
    if isfield(warp,'R_mat'); warp.trinfo.R_mat = warp.R_mat; end
    if isfield(warp,'scale'); warp.trinfo.scale = warp.scale; end
    input_warping = 1;
else
    input_warping = 0;
end

if input_warping
    X_prior = warpvars(X,'dir',warp.trinfo);
    y_prior = y + warpvars(X_prior,'logpdf',warp.trinfo);
else
    X_prior = X;
    y_prior = y;
end

Ncov = D+1;     % Number of covariance function hyperparameters

% Get mean function hyperparameter info
[Nmean,meaninfo] = gplite_meanfun([],X_prior,meanfun,y_prior);

if isempty(hyp0); hyp0 = zeros(Ncov+Nmean+1,1); end
[Nhyp,N0] = size(hyp0);      % Hyperparameters

LB = [];
UB = [];
if isfield(hprior,'LB'); LB = hprior.LB; end
if isfield(hprior,'UB'); UB = hprior.UB; end
if isempty(LB); LB = NaN(1,Nhyp); end
if isempty(UB); UB = NaN(1,Nhyp); end
LB = LB(:)'; UB = UB(:)';

if ~isfield(hprior,'mu') || isempty(hprior.mu)
    hprior.mu = NaN(Nhyp,1);
end
if ~isfield(hprior,'sigma') || isempty(hprior.sigma)
    hprior.sigma = NaN(Nhyp,1);
end
if ~isfield(hprior,'df') || isempty(hprior.df)
    hprior.df = DfBase*ones(Nhyp,1);
end
if numel(hprior.mu) < Nhyp; hprior.mu = [hprior.mu(:); NaN(Nhyp-numel(hprior.mu),1)]; end
if numel(hprior.sigma) < Nhyp; hprior.sigma = [hprior.sigma(:); NaN(Nhyp-numel(hprior.sigma),1)]; end
if numel(hprior.df) < Nhyp; hprior.df = [hprior.df(:); NaN(Nhyp-numel(hprior.df),1)]; end


% Default hyperparameter lower and upper bounds, if not specified
width = max(X_prior) - min(X_prior);
height = max(y_prior)-min(y_prior);


% Read hyperparameter bounds, if specified; otherwise set defaults
LB_ell = LB(1:D);   
idx = isnan(LB_ell);                 LB_ell(idx) = log(width(idx))+log(ToL);
LB_sf = LB(D+1);        if isnan(LB_sf); LB_sf = log(height)+log(ToL); end
LB_sn = LB(Ncov+1);     if isnan(LB_sn); LB_sn = log(ToL); end

% Set mean function hyperparameters lower bounds
LB_mean = LB(Ncov+2:D+2+Nmean);
idx = isnan(LB_mean);
LB_mean(idx) = meaninfo.LB(idx);

LB_alpha = -5*ones(1,D);
LB_beta = -5*ones(1,D);
if ~input_warping; LB_alpha = []; LB_beta = []; end

UB_ell = UB(1:D);   
idx = isnan(UB_ell);    UB_ell(idx) = log(width(idx)*10);
UB_sf = UB(D+1);        if isnan(UB_sf); UB_sf = log(height*10); end
UB_sn = UB(Ncov+1);     if isnan(UB_sn); UB_sn = log(height); end

% Set mean function hyperparameters upper bounds
UB_mean = UB(Ncov+2:D+2+Nmean);
idx = isnan(UB_mean);
UB_mean(idx) = meaninfo.UB(idx);

UB_alpha = 5*ones(1,D);
UB_beta = 5*ones(1,D);
if ~input_warping; UB_alpha = []; UB_beta = []; end

% Create lower and upper bounds
LB = [LB_ell,LB_sf,LB_sn,LB_mean,LB_alpha,LB_beta];
UB = [UB_ell,UB_sf,UB_sn,UB_mean,UB_alpha,UB_beta];
UB = max(LB,UB);

% Plausible bounds for generation of starting points
PLB_ell = log(width)+0.5*log(ToL);
PUB_ell = log(width);

PLB_sf = log(height)+0.5*log(ToL);
PUB_sf = log(height);

PLB_sn = 0.5*log(ToL);
PUB_sn = log(std(y_prior));

PLB_mean = meaninfo.PLB;
PUB_mean = meaninfo.PUB;

PLB_alpha = -2*ones(1,D);
PUB_alpha = 2*ones(1,D);

PLB_beta = -2*ones(1,D);
PUB_beta = 2*ones(1,D);

if ~input_warping; PLB_alpha = []; PUB_alpha = []; PLB_beta = []; PUB_beta = []; end

PLB = [PLB_ell,PLB_sf,PLB_sn,PLB_mean,PLB_alpha,PLB_beta];
PUB = [PUB_ell,PUB_sf,PUB_sn,PUB_mean,PUB_alpha,PUB_beta];

PLB = min(max(PLB,LB),UB);
PUB = max(min(PUB,UB),LB);

if input_warping; gradobj = 'on'; else; gradobj = 'on'; end
gptrain_options = optimoptions('fmincon','GradObj',gradobj,'Display','off');    

%% Hyperparameter optimization
if Ns > 0 && ~strcmpi(Sampler,'laplace')
    gptrain_options.OptimalityTolerance = 0.1;  % Limited optimization
else
    gptrain_options.OptimalityTolerance = 1e-6;        
end

hyp = zeros(Nhyp,Nopts);
nll = [];

% Initialize GP
if input_warping
    gp.X = X;   % X and y must be in pre-warping coordinate space
    gp.y = y;
    gp.post = [];
    gp.hyp = hyp0(:,1);
    gp.warp = warp;
    gp.meanfun = meanfun;
    gp = gp_objfun(hyp0(:,1),gp,[],1,0);
    gp.X = X; gp.y = y;
else
    gp = gplite_post(hyp0(:,1),X,y,meanfun);
end

forcepriors = strcmpi(Sampler,'laplace');
if forcepriors
    for i = 1:numel(hprior.mu)
        if ~isfinite(hprior.mu(i)) || ~isfinite(hprior.sigma(i))
            hprior.mu(i) = 0.5*(PLB(i) + PUB(i));
            hprior.sigma(i) = 0.5*(PUB(i)-PLB(i));
            hprior.df(i) = DfBase;
        end
        % hprior.sigma(i) = hprior.sigma(i)/1000;
        % Remove box constraints
        LB(i) = -Inf;   UB(i) = Inf;
    end
end

% Define objective functions for optimization
gpoptimize_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,0);

% Compare old probability with new probability, check amount of change
if ~isempty(LogP) && Ns > 0
    nll = Inf(1,size(hyp0,2));
    for i = 1:size(hyp0,2); nll(i) = gpoptimize_fun(hyp0(:,i)); end    
    lnw = -nll - LogP(:)';
    w = exp(lnw - max(lnw));
    w = w/sum(w);
    ESS_frac = (1/sum(w.^2))/size(hyp0,2);

    ESS_thresh = 0.5;
    % Little change, keep sampling
    if ESS_frac > ESS_thresh && strcmpi(Sampler,'slicelite')
        Ninit = 0;
        Nopts = 0;
        if strcmpi(Sampler,'slicelite')
            Thin_eff = max(1,round(Thin*(1 - (ESS_frac-ESS_thresh)/(1-ESS_thresh))));
            Burnin = Thin_eff*Ns;
            Thin = 1;
        end
    end
end

% First evaluate GP log posterior on an informed space-filling design
if Ninit > 0
    optfill.FunEvals = Ninit;
    [~,~,~,output_fill] = fminfill(gpoptimize_fun,hyp0',LB,UB,PLB,PUB,hprior,optfill);
    hyp(:,:) = output_fill.X(1:Nopts,:)';
    widths_default = std(output_fill.X,[],1);
else
    if isempty(nll)
        nll = Inf(1,size(hyp0,2));
        for i = 1:size(hyp0,2); nll(i) = gpoptimize_fun(hyp0(:,i)); end
    end
    [nll,ord] = sort(nll,'ascend');
    hyp = hyp0(:,ord);
    widths_default = PUB - PLB;
end

% if input_warping
%     derivcheck(gpoptimize_fun,hyp0(:,1),1);
% end

% Check that hyperparameters are within bounds
hyp = bsxfun(@min,UB'-eps(UB'),bsxfun(@max,LB'+eps(LB'),hyp));

%tic
% Perform optimization from most promising NOPTS hyperparameter vectors
for iTrain = 1:Nopts
    nll = Inf(1,Nopts);
    try
        [hyp(:,iTrain),nll(iTrain)] = ...
            fmincon(gpoptimize_fun,hyp(:,iTrain),[],[],[],[],LB,UB,[],gptrain_options);
    catch
        % Could not optimize, keep starting point
    end
end
%toc

[~,idx] = min(nll); % Take best hyperparameter vector
hyp_start = hyp(:,idx);

% Check that starting point is inside current bounds
hyp_start = min(max(hyp_start',LB+eps(LB)),UB-eps(UB))';

logp_prethin = [];  % Log posterior of samples

%% Sample from best hyperparameter vector using slice sampling
if Ns > 0

    Ns_eff = Ns*Thin;   % Effective number of samples (thin after)

    switch lower(Sampler)
        case 'slicesample'
            gpsample_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,1);
            sampleopts.Thin = 1;
            sampleopts.Burnin = Burnin;
            sampleopts.Display = 'off';
            sampleopts.Diagnostics = false;
            if isempty(Widths)
                Widths = widths_default;
            else
                Widths = min(Widths(:)',widths_default);
                % [Widths; widths_default]
            end

            [samples,fvals,exitflag,output] = ...
                slicesamplebnd(gpsample_fun,hyp_start',Ns_eff,Widths,LB,UB,sampleopts);
            hyp_prethin = samples';
            logp_prethin = fvals;

        case 'slicelite'
            gpsample_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,1);
            sampleopts.Thin = 1;
            sampleopts.Burnin = Burnin;
            sampleopts.Display = 'off';
            if isempty(Widths)
                Widths = widths_default;
            else
                Widths = min(Widths(:)',widths_default);
                % [Widths; widths_default]
            end



            try
            if Nopts == 0
                sampleopts.Adaptive = false;
                if size(hyp,2) < Ns_eff
                    hyp = repmat(hyp,[1,ceil(Ns_eff/size(hyp,2))]);
                    hyp = hyp(:,1:Ns_eff);
                end                
                [samples,fvals,exitflag,output] = ...
                    slicelite(gpsample_fun,hyp',Ns_eff,Widths,LB,UB,sampleopts);                
            else            
                sampleopts.Adaptive = true;
                [samples,fvals,exitflag,output] = ...
                    slicelite(gpsample_fun,hyp_start',Ns_eff,Widths,LB,UB,sampleopts);
            end
            catch
                pause
            end
            hyp_prethin = samples';
            logp_prethin = fvals;

        case 'covsample'
            gpsample_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,1);            
            sampleopts.Thin = 1;
            sampleopts.Burnin = Burnin;
            sampleopts.Display = 'off';
            sampleopts.Diagnostics = false;
            sampleopts.VarTransform = false;
            sampleopts.InversionSample = false;
            sampleopts.FitGMM = false;
            sampleopts.TolX = 1e-80;
            sampleopts.WarmUpStages = 1;
            W = 1;

            samples = ...
                eissample_lite(gpsample_fun,hyp_start',Ns_eff,W,Widths,LB,UB,sampleopts);
            hyp_prethin = samples';            

        case 'hmc'            
            gpsample_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,0);
            sampleopts.display = 0;
            sampleopts.checkgrad = 0;
            sampleopts.steps = 10;
            sampleopts.nsamples = Ns_eff;
            sampleopts.stepadj = 0.01;
            sampleopts.widths = [];
            sampleopts.nomit = Burnin;
            sampleopts.widths = Widths;

            [samples,fvals,diagn] = ...
                hmc2(gpsample_fun,hyp_start',sampleopts,@(hyp) gpgrad_fun(hyp,gpsample_fun));            
            hyp_prethin = samples';

        case 'laplace'
            hyp_mode = hyp_start;
            gpsample_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,1);
            Hess = grad2hess(gpsample_fun,hyp_mode');
            Sigma = inv(-Hess);
            diag(Sigma)'
            [cholsigma,p] = chol(Sigma);
            if p > 0
                % Failed Cholesky decomposition, use only MAP
                hyp = hyp_mode;
                [hyp';LB;UB]
            else
                hyp = bsxfun(@plus,hyp_mode,cholsigma'*randn(Nhyp,Ns_eff));
                for i = 1:Ns
                    y = gpsample_fun(hyp(i,:));
                    y2 = mvnlogpdf(hyp(i,:),hyp_mode,Sigma);
                    iw(i) = y/y2;
                end
                iw
            end            
            hyp_prethin = bsxfun(@min,bsxfun(@max,hyp,LB(:)),UB(:));

%           case 'nuts'            
%     nuts_opt.M = Ns*sampleopts.Thin;
%     nuts_opt.Madapt= 10*Ns;
%     [hyp_nuts,fvals_nuts,diagn_nuts] = hmc_nuts(@gpsample_fun,hyp(:,idx)',nuts_opt);

        otherwise
            error('gplite_train:UnknownSampler', ...
                'Unknown MCMC sampler for GP hyperparameters.');
    end

    % Thin samples
    hyp = hyp_prethin(:,Thin:Thin:end);
    logp = logp_prethin(Thin:Thin:end);

else
    hyp = hyp(:,idx);
    hyp_prethin = hyp;
    logp_prethin = -nll;
    logp = -nll(idx);
end

% Recompute GP with finalized hyperparameters
gp = gp_objfun(hyp,gp,[],1);

% Additional OUTPUT struct
if nargout > 2
    output.LB = LB;
    output.UB = UB;
    output.PLB = PLB;
    output.PUB = PUB;
    output.hyp_prethin = hyp_prethin;
    output.logp = logp;
    output.logp_prethin = logp_prethin;
end


% Check GP posteriors
% for s = 1:numel(gp.post)
%     if ~all(isfinite(gp.post(s).L(:)))
%         pause
%     end
% end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dnlZ = gpgrad_fun(hyp,gpsample_fun)
    [~,dnlZ] = gpsample_fun(hyp);
    dnlZ = dnlZ';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nlZ,dnlZ] = gp_objfun(hyp,gp,hprior,gpflag,swapsign)
%GPLITE_OBJFUN Objective function for hyperparameter training.

if nargin < 5 || isempty(swapsign); swapsign = 0; end

compute_grad = nargout > 1 && ~gpflag;
[N,D] = size(gp.X);

if isfield(gp,'warp') && ~isempty(gp.warp)
    Nwarp = gp.warp.Nwarp;
    hyp_warp = hyp(end-Nwarp+1:end);          % Warping parameters

    if Nwarp > 0
        alpha = exp(hyp_warp(1:D));
        beta = exp(hyp_warp(D+(1:D)));
        gp.warp.trinfo.alpha = alpha(:)';
        gp.warp.trinfo.beta = beta(:)';
    end

    gp.X = warpvars(gp.X,'dir',gp.warp.trinfo);
    if gp.warp.logpdf_flag
        % The function is a log pdf, so apply log Jacobian correction
        gp.y = gp.y + warpvars(gp.X,'logp',gp.warp.trinfo);
    end
    warp = gp.warp;
else
    Nwarp = 0;
    warp = [];
end

if gpflag
    gp = gplite_post(hyp(1:end-Nwarp,:),gp.X,gp.y,gp.meanfun);
    if ~isempty(warp); gp.warp = warp; end
    nlZ = gp;
else

    try
        % Compute negative log marginal likelihood (without prior)
        if compute_grad && Nwarp > 0
            if ~isempty(warp.trinfo.R_mat); R_mat = warp.trinfo.R_mat; else; R_mat = eye(D); end
            if ~isempty(warp.trinfo.scale); scale = warp.trinfo.scale; else; scale = ones(1,D); end        

            [nlZ,dnlZ,post,K_mat,Q] = gplite_nlZ(hyp(1:end-Nwarp,:),gp,[]);

            % Get gradient of warpings (ignores roto-scaling)
            dg_inv = warpvars(gp.X,'g',gp.warp.trinfo);

            % Get derivatives wrt warping parameters for warping and its derivative
            dgdtheta = warpvars(gp.X,'f',gp.warp.trinfo);
            dgprimedtheta = warpvars(gp.X,'m',gp.warp.trinfo);

            % Compute gradient of warping parameters
            ell2 = exp(2*hyp(1:D));

            dnlZ_warp = zeros(Nwarp,1);

            XX = zeros(N,N,D);  % Precompute matrix of cross-differences
            for j = 1:D
                XX(:,:,j) = bsxfun(@minus,gp.X(:,j)/ell2(j),gp.X(:,j)'/ell2(j));
            end

            for k = 1:D
                dxi_dalpha = bsxfun(@times,(1./scale .* R_mat(k,:)), dgdtheta(:,k));
                dxi_dbeta = bsxfun(@times,(1./scale .* R_mat(k,:)), dgdtheta(:,k+D));

                % Gradients due to warping of the covariance matrix
                K_temp = zeros(size(K_mat));
                for j = 1:D
                    K_temp = K_temp + XX(:,:,j).*bsxfun(@minus,dxi_dalpha(:,j),dxi_dalpha(:,j)');
                end
                K_temp = -K_temp .* K_mat;
                dnlZ_warp(k) = dnlZ_warp(k) + sum(sum(Q.*K_temp))/2;

                K_temp = zeros(size(K_mat));
                for j = 1:D
                    K_temp = K_temp + XX(:,:,j).*bsxfun(@minus,dxi_dbeta(:,j),dxi_dbeta(:,j)');
                end
                K_temp = -K_temp .* K_mat;
                dnlZ_warp(k+D) = dnlZ_warp(k+D) + sum(sum(Q.*K_temp))/2;

                % Gradients due to warping of the mean function
                switch gp.meanfun
                    case {0,1}

                    case 4
                        hyp_mean = hyp(gp.Ncov+2:gp.Ncov+1+gp.Nmean); % Get mean function hyperparameters
                        xm = hyp_mean(1+(1:D))';
                        omega = exp(hyp_mean(D+1+(1:D)))';
                        dm = -bsxfun(@rdivide,bsxfun(@minus,gp.X,xm), omega.^2);

                        dnlZ_warp(k) = dnlZ_warp(k) - sum(dm.*dxi_dalpha,2)'*post.alpha;
                        dnlZ_warp(k+D) = dnlZ_warp(k+D) - sum(dm.*dxi_dbeta,2)'*post.alpha;

                    otherwise
                        error('gplite_train:WarpedMeanFun', ...
                            'Unsupported mean function for input warping.');
                end
            end

            % Gradients due to warping of the function value
            if gp.warp.logpdf_flag
                dnlZ_warp(:) = dnlZ_warp(:) - (dgprimedtheta(:,:).*repmat(dg_inv(:,:),[1 2]))'*post.alpha;
            end

            % Jacobian correction for log representation of warping hyperparameters
            dnlZ_warp = dnlZ_warp .* exp(hyp_warp(:));

            % Add warping parameters gradients
            dnlZ = [dnlZ; dnlZ_warp];

        elseif compute_grad
            [nlZ,dnlZ] = gplite_nlZ(hyp(1:end-Nwarp,:),gp,[]);
        else
            nlZ = gplite_nlZ(hyp(1:end-Nwarp,:),gp,[]);
        end

        % Add log prior if present, with all parameters
        if ~isempty(hprior)
            if compute_grad
                [P,dP] = gplite_hypprior(hyp,hprior);
                nlZ = nlZ - P;
                dnlZ = dnlZ - dP;
            else
                P = gplite_hypprior(hyp,hprior);
                nlZ = nlZ - P;
            end
        end

        % Swap sign of negative log marginal likelihood (e.g., for sampling)
        if swapsign
            nlZ = -nlZ;
            if compute_grad; dnlZ = -dnlZ; end
        end

    catch
        % Something went wrong, return NaN but try to continue
        nlZ = NaN;
        dnlZ = NaN(size(hyp));        
    end

%     if compute_grad
%         dnlZ
%     end

end

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function gp = gplite_post(hyp,X,y,meanfun,update1)
%GPLITE_POST Compute posterior GP for a given training set.
%   GP = GPLITE_POST(HYP,X,Y,MEANFUN) computes the posterior GP for a vector
%   of hyperparameters HYP and a given training set. HYP is a column vector
%   of hyperparameters (see below). X is a N-by-D matrix of training inputs
%   and Y a N-by-1 vector of training targets (function values at X).
%   MEANFUN is the GP mean function (see GPLITE_MEANFUN.M for a list).
%
%   GP = GPLITE_POST(GP,XSTAR,YSTAR,[],1) performs a fast rank-1 update for
%   a GPLITE structure, given a single new observation at XSTAR with observed
%   value YSTAR.
%
%   Note that the returned GP contains auxiliary structs for faster
%   computations. To save memory, call GPLITE_CLEAN.
%
%   See also GPLITE_CLEAN, GPLITE_MEANFUN.

if nargin < 5 || isempty(update1); update1 = false; end

gp = [];
if isstruct(hyp)
    gp = hyp;
    if nargin < 2; X = gp.X; end
    if nargin < 3; y = gp.y; end
    if nargin < 4; meanfun = gp.meanfun; end
end

if update1
    if size(X,1) > 1
        error('gplite_post:NotRankOne', ...
          'GPLITE_POST with this input format only supports rank-one updates.');
    end
    if isempty(gp)
        error('gplite_post:NoGP', ...
          'GPLITE_POST can perform rank-one update only with an existing GP struct.');        
    end
    if ~isempty(meanfun)
        warning('gplite_post:RankOneMeanFunction', ...
            'No need to specify a GP mean function when performin a rank-one update.');
    end
end

% Create GP struct
if isempty(gp)
    gp.X = X;
    gp.y = y;
    gp.post = [];
    [N,D] = size(X);            % Number of training points and dimension
    [Nhyp,Ns] = size(hyp);      % Number of hyperparameters and samples
    gp.meanfun = meanfun;    
    for s = 1:size(hyp,2); gp.post(s).hyp = hyp(:,s); end
    gp.Ncov = D+1;                 % Number of covariance function hyperparameters
    gp.Nmean = gplite_meanfun([],X,meanfun);
    if Nhyp ~= gp.Ncov+1+gp.Nmean
        error('gplite_post:dimmismatch', ...
            'Number of hyperparameters mismatched with dimension of training inputs.');
    end
else
    [N,D] = size(gp.X);         % Number of training points and dimension
    Ns = numel(gp.post);        % Hyperparameter samples
end

Ncov = gp.Ncov;
Nmean = gp.Nmean;


if ~update1

    % Loop over hyperparameter samples
    for s = 1:Ns
        hyp = gp.post(s).hyp;

        % Extract GP hyperparameters from HYP
        ell = exp(hyp(1:D));
        sf2 = exp(2*hyp(D+1));
        sn2 = exp(2*hyp(Ncov+1));
        sn2_mult = 1;  % Effective noise variance multiplier

        % Evaluate mean function on training inputs
        hyp_mean = hyp(Ncov+2:Ncov+1+Nmean); % Get mean function hyperparameters        
        m = gplite_meanfun(hyp_mean,X,gp.meanfun);

        % Compute kernel matrix K_mat
        K_mat = sq_dist(diag(1./ell)*X');
        K_mat = sf2 * exp(-K_mat/2);

        if sn2 < 1e-6   % Different representation depending on noise size
            for iter = 1:10     % Cholesky decomposition until it works
                [L,p] = chol(K_mat+sn2*sn2_mult*eye(N));
                if p > 0; sn2_mult = sn2_mult*10; else; break; end
            end
            sl = 1;
            pL = -L\(L'\eye(N));    % L = -inv(K+inv(sW^2))
            Lchol = 0;         % Tiny noise representation
        else

            for iter = 1:10
                [L,p] = chol(K_mat/(sn2*sn2_mult)+eye(N));
                if p > 0; sn2_mult = sn2_mult*10; else; break; end
            end
            sl = sn2*sn2_mult;
            pL = L;                 % L = chol(eye(n)+sW*sW'.*K)
            Lchol = 1;
        end

        alpha = L\(L'\(y-m)) ./ sl;     % alpha = inv(K_mat + sn2.*eye(N)) * (y - m)

        % GP posterior parameters
        gp.post(s).alpha = alpha;
        gp.post(s).sW = ones(N,1)/sqrt(sn2*sn2_mult);   % sqrt of noise precision vector
        gp.post(s).L = pL;
        gp.post(s).sn2_mult = sn2_mult;
        gp.post(s).Lchol = Lchol;
    end

else
    % Added training input
    xstar = X;
    ystar = y;

    % Rank-1 update for the same XSTAR but different ystars
    Nstar = numel(ystar);

    if Nstar > 1; gp(2:Nstar) = gp(1); end

    % Compute prediction for all samples
    [mstar,vstar] = gplite_pred(gp(1),xstar,[],1);

    % Loop over hyperparameter samples
    for s = 1:Ns

        hyp = gp(1).post(s).hyp;

        % Extract GP hyperparameters from HYP
        ell = exp(hyp(1:D));
        sf2 = exp(2*hyp(D+1));
        sn2 = exp(2*hyp(Ncov+1));
        sn2_eff = sn2*gp(1).post(s).sn2_mult;            

        % Compute covariance and cross-covariance
        K = sf2;
        Ks_mat = sq_dist(diag(1./ell)*gp(1).X',diag(1./ell)*xstar');
        Ks_mat = sf2 * exp(-Ks_mat/2);    

        L = gp(1).post(s).L;
        Lchol = gp(1).post(s).Lchol;

        if Lchol        % high-noise parameterization
            alpha_update = (L\(L'\Ks_mat)) / sn2_eff;
            new_L_column = linsolve(L, Ks_mat, ...
                                struct('UT', true, 'TRANSA', true)) / sn2_eff;
            gp(1).post(s).L = [L, new_L_column; ...
                               zeros(1, size(L, 1)), ...
                               sqrt(1 + K / sn2_eff - new_L_column' * new_L_column)];
        else            % low-noise parameterization
            alpha_update = -L * Ks_mat;
            v = -alpha_update / vstar(:,s);
            gp(1).post(s).L = [L + v * alpha_update', -v; -v', -1 / vstar(:,s)];
        end

        gp(1).post(s).sW = [gp(1).post(s).sW; gp(1).post(s).sW(1)];

        for iStar = 2:Nstar
            gp(iStar).post(s) = gp(1).post(s);
        end

        % alpha_update now contains (K + \sigma^2 I) \ k*
        for iStar = 1:Nstar
            gp(iStar).post(s).alpha = ...
                [gp(iStar).post(s).alpha; 0] + ...
                (mstar(:,s) - ystar(iStar)) / vstar(:,s) * [alpha_update; -1];
        end
    end

    % Add single input to training set
    for iStar = 1:Nstar
        gp(iStar).X = [gp(iStar).X; xstar];
        gp(iStar).y = [gp(iStar).y; ystar(iStar)];
    end
end






```
