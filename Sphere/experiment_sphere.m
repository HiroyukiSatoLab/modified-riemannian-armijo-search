% Rayleigh quotient minimization on the sphere
% Comparison between the existing and proposed methods.

eps = 1e-4; % Convergence tolerance: stop when the gradient norm is smaller than eps.
beta = .5;  % Contraction rate for the backtracking.
tau = 1e-4; % Parameter for the Armijo condition.
flag = 0;   % Logging flag: 0 = no log output, 1 = with log output.
sizeM = 5;  % Number of test problem instances.

% Arrays for recording computation times of each method.
time1 = zeros(1, sizeM);   % Elapsed time for the existing method.
time2 = zeros(1, sizeM);   % Elapsed time for the proposed method.

% Arrays for recording iteration counts.
k1 = zeros(1,sizeM);
k2 = zeros(1,sizeM);

for size = 1:sizeM
    n = 400 * size; % Matrix size.
    fprintf("n = %d\n", n);

    % Run both methods and record the results.
    [k1(size), ~, time1(size)] = func_SD_sphere(n, eps, beta, tau, flag);
    [k2(size), ~, time2(size)] = func_proposedSD_sphere(n, eps, beta, tau, flag);
end

% Plot the computation times for comparison.
hor = 400 * (1:sizeM);
plot(hor, time1, 'o-', hor, time2, 'x-');
xlabel("$$n$$", "Interpreter","latex");
ylabel("Time [s]");
ax = gca;
ax.FontSize = 18;
xlim([hor(1), hor(end)]);
ylim([0, 400]);
xticks(400:400:2000)
legend("Existing method", "Proposed method", 'Location','northwest');
f = gcf;