%% Riemannian steepest descent with modified Armijo backtracking on the SPD manifold


function [k, bt, time] = func_proposedSD_SPD(n, eps, beta, tau, flag)
% Inputs:
%   n    : problem size
%   eps  : stopping tolerance; terminate when the Riemannian gradient norm < eps
%   beta : contraction factor for backtracking
%   tau  : parameter in the Armijo condition
%   flag : logging option (1 = display progress, 0 = silent)
%
% Outputs:
%   k     : number of outer iterations
%   bt    : number of backtracking iterations at each outer iterate
%   time  : total elapsed time in seconds

symm = @(W) .5 * (W+W');   % Symmetric part of W

% Parameter setting
rng(0);                    % Fix the random seed for reproducibility
M = sympositivedefinitefactory(n);
M.inner = @(X, eta, zeta) eta(:)' * zeta(:);
M.norm = @(X, eta) norm(eta,'fro');
M.egrad2rgrad = @egrad2rgrad;
function eta = egrad2rgrad(X, eta)
    eta = symm(eta);
end
X0 = rand(n) - .5 * ones(n);
X0 = .5 * (X0 + X0');
X0 = eye(n) + .001 * X0;   % initial point on the SPD manifold

% Modified Armijo search + steepest descent
tic;
X = X0;
D = det(X);
f = (D - 1)^2;             % Objective function value
Rgrad = 2 * D * (D-1) * X; % Riemannian gradient (negative gradient)
k = 0;
bt = zeros(1, 10000);      % Backtracking iteration count at each outer iteration
retcnt = zeros(1, 10000);  % Retraction count at each outer iteration
l = 0;                     % Counter for backtracking
r = 0;                     % Counter for retraction
while M.norm(X, Rgrad) >= eps
    if flag == 1
        fprintf("k: %3d, Func. val.: %f, Grad. norm: %f, bt： %d\n, retcnt: %d\n", k, f, M.norm(X,Rgrad), l, r);
    end
    p = -Rgrad;                % Search direction (negative gradient)
    gp = M.inner(X, Rgrad, p); % Inner product <g, p>
    t = 1.0;                   % Initial step length candidate
    l = 0;
    r = 0;
    while 1
        tp = t * p;
        Xtp = X + tp;          % Point on a line (outside the SDP manifold)
        DXtp = det(Xtp);
        fXtp = (DXtp-1)^2;     % f(X+tp)
        invXp = X \ p;

        % Check the approximate Armijo condition
        if fXtp <= f + tau * t * gp
            Xretr = symm(X * expm(t*invXp)); % Retraction
            r = r+1;
            DXretr = det(Xretr);
            fX = (DXretr-1)^2;               % f(R_X(tp))
            if fX <= f + tau * t * gp
                X = Xretr;
                break
            end
        end
        t = t * beta;          % Contract step size if Armijo condition not satisfied
        l = l+1;
    end
    bt(k+1) = l;
    retcnt(k+1) = r;
    f = fX;                    % Update objective value
    Rgrad = 2*DXretr*(DXretr-1)*X;
    k = k + 1;
end
time = toc;
fprintf("Proposed method:\n");
fprintf("Time：%f sec., Iter. %d, bt: %d, retcnt: %d\n", time, k, sum(bt), sum(retcnt));
end
