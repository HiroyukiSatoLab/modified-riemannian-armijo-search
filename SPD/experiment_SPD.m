% minimization on SPD
% Comparison between the existing and proposed methods.

eps = 1e-4; % % Convergence tolerance: stop when the gradient norm is smaller than eps.
beta = .5; % Contraction rate for the backtracking.
tau = 1e-4; % Parameter for the Armijo condition.
flag = 0; % Logging flag: 0 = no log output, 1 = with log output.
sizeM = 5; % Number of test problem instances.

% Arrays for recording computation times of each method.
time1 = zeros(1, sizeM);   % Elapsed time for the existing method.
time2 = zeros(1, sizeM);   % Elapsed time for the proposed method.

% Arrays for recording iteration counts.
k1 = zeros(1,sizeM);
k2 = zeros(1,sizeM);

for size = 1:sizeM
    n = 200 * size; % Matrix size.
    fprintf("n = %d\n", n);

    % Run both methods and record the results.
    [k1(size), ~, time1(size)] = func_SD_SPD(n, eps, beta, tau, flag);
    [k2(size), ~, time2(size)] = func_proposedSD_SPD(n, eps, beta, tau, flag);
    fprintf("\n");
end

% Plot the computation times for comparison.
hor = 200 * (1:sizeM);
plot(hor, time1, 'o--', hor, time2, 'x--');
xlabel("n");
ylabel("Time [s]");
xlim([0, hor(end)]);
ylim([0, max([max(time1), max(time2)])+1]);