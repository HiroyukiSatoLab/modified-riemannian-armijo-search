%% Riemannian steepest descent with Armijo backtracking on the SPD manifold

function [k, bt, time] = func_SD_SPD(n, eps, beta, tau, flag)
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

% Steepest descent method with Riemannian Armijo backtracking
tic;
X = X0;
D = det(X);
f = (D - 1)^2;             % Objective function value
Rgrad = 2 * D * (D-1) * X; % Riemannian gradient
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
    invXp = X \ p;
    while 1
        Xp = symm(X * expm(t*invXp)); % Retraction
        r = r+1;
        Dp = det(Xp);
        fXp = (Dp-1)^2;               % f(R_x(tp))

        % Check the Armijo condition
        if fXp <= f + tau * t * gp
            X = Xp;
            D = Dp;
            break;
        end
        t = t * beta;                 % Contract step size if Armijo condition not satisfied
        l = l+1;
    end
    bt(k+1) = l;
    retcnt(k+1) = r;
    f = fXp;                   % Update objective value
    Rgrad = 2*D*(D-1)*X;
    k = k + 1;
end
time = toc;
fprintf("Existing method:\n");
fprintf("Time：%f sec., Iter. %d, bt: %d, retcnt: %d\n", time, k, sum(bt), sum(retcnt));
end
