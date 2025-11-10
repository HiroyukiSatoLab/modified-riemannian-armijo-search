%% Riemannian steepest descent with modified Armijo backtracking on the Stiefel manifold

function [k, bt, time] = func_proposedSD_Stiefel(n, sizer, eps, beta, tau, flag)
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

symm = @(W) .5*(W+W');        % Symmetric part of W

% Parameter setting
rng(0);                       % Fix the random seed for reproducibility
M = stiefelfactory(n,sizer);
X0 = M.rand();                % Initial point on the Stiefel manifold

% Symmetric matrix for defining the objective function
A = rand(n);
A = .5 * (A+A');
N = diag(sizer:-1:1);

% Modified Armijo search + steepest descent
tic;
X = X0;
AXN = A * X * N;
XAXN = X'*AXN;
f = .5 * trace(XAXN);         % Objective function value
Rgrad = AXN - X * symm(XAXN); % Riemannian gradient
k = 0;
bt = zeros(1,10000);          % Backtracking iteration count at each outer iteration
retcnt = zeros(1,10000);      % Retraction count at each outer iteration
l = 0;                        % Counter for backtracking
r = 0;                        % Counter for retraction
while norm(Rgrad, 'fro') >= eps
    if flag == 1
        fprintf("k: %3d, Func. val.: %f, Grad. norm: %f, bt： %d\n, retcnt: %d\n", k, f, M.norm(X,Rgrad), l, r);
    end
    p = -Rgrad;               % Search diretion (negative gradient)
    gp = Rgrad(:)' * p(:);    % Inner product <g, p>
    t = 1.0;                  % Initial step length candidate
    l = 0;
    r = 0;
    while 1
        Xtp = X + t*p;                  % Point on a line (outside the Stiefel manifold)
        AXtpN = A*Xtp*N;
        fXtp = .5 * Xtp(:)' * AXtpN(:); % f(X+tp)

        % Check the approximate Armijo condition        
        if fXtp <= f + tau * t * gp 
            Xretr = qr_unique(Xtp);           % Retraction based on QR decomposition
            r = r+1;
            AXretrN = A*Xretr*N;
            fX = .5 * Xretr(:)' * AXretrN(:); % f(R_x(tp))
            if fX <= f + tau * t * gp
                X = Xretr;
                break
            end
        end
        t = t * beta;                   % Contract step size if Armijo condition not satisfied
        l = l+1;
    end
    bt(k+1) = l;
    retcnt(k+1) = r;
    f = fX;                   % Update objective value
    Rgrad = AXretrN - X * symm(X'*AXretrN);
    k = k + 1;
end
time = toc;
fprintf("Proposed method:\n");
fprintf("Time：%f sec., Iter. %d, bt: %d, retcnt: %d\n", time, k, sum(bt), sum(retcnt));
end
