%% Riemannian steepest descent with modified Armijo backtracking on the sphere

function [k, bt, time] = func_proposedSD_sphere(n, eps, beta, tau, flag)
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

% Parameter setting
rng(0);                   % Fix the random seed for reproducibility
M = spherefactory(n);
x0 = M.rand();            % Initial point on the sphere

% Symmetric matrix defining the objective function
A = rand(n);
A = .5 * (A+A');

% Modified Armijo search + steepest descent
tic;
x = x0;
Ax = A * x;
xAx = x'*Ax;
f = .5 * xAx;             % Objective function value
Rgrad = Ax - xAx * x;     % Riemannian gradient
k = 0;
bt = zeros(1, 10000);     % Backtracking iteration count at each outer iteration
retcnt = zeros(1, 10000); % Retraction count at each outer iteration
l = 0;                    % Counter for backtracking
r = 0;                    % Counter for retraction
while norm(Rgrad) >= eps
    if flag == 1
        fprintf("k: %3d, Func. val.: %f, Grad. norm: %f, bt： %d\n, retcnt: %d\n", k, f, M.norm(x,Rgrad), l, r);
    end
    p = -Rgrad;           % Search diretion (negative gradient)
    gp = Rgrad' * p;      % Inner product <g, p>
    t = 1.0;              % Initial step length candidate
    l = 0;
    r = 0;
    while 1
        xtp = x + t*p;              % Point on a line (outside the sphere)
        fxtp = .5 * xtp' * A * xtp; % f(x+tp)

        % Check the approximate Armijo condition
        if fxtp <= f + tau * t * gp
            xretr = xtp/norm(xtp);  % Retraction by normalization
            r = r + 1;
            Ax = A*xretr;
            xAx = xretr'*Ax;
            fx = .5 * xAx;          % f(R_x(tp))
            if fx <= f + tau * t * gp
                x = xretr;
                break
            end
        end
        t = t * beta;               % Contract step size if Armijo condition not satisfied
        l = l + 1;
    end
    bt(k+1) = l;
    retcnt(k+1) = r;
    f = fx;               % Update objective value
    Rgrad = Ax - xAx * x;
    k = k + 1;
end
time = toc;
fprintf("Proposed method:\n");
fprintf("Time：%f sec., Iter. %d, bt: %d, retcnt: %d\n", time, k, sum(bt), sum(retcnt));
end