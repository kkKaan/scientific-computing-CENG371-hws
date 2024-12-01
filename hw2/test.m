max_n = 30; % Maximum size of Hilbert matrices
n_values = 1:max_n;

% Arrays to store timing and relative errors
time_sherman = zeros(size(n_values));
rel_error_sherman = zeros(size(n_values));

time_pickett = zeros(size(n_values));
rel_error_pickett = zeros(size(n_values));

time_crout = zeros(size(n_values));
rel_error_crout = zeros(size(n_values));

time_builtin = zeros(size(n_values));
rel_error_builtin = zeros(size(n_values));

% Loop over matrix sizes
for i = 1:length(n_values)
    n = n_values(i);
    A = hilb(n); % Generate the Hilbert matrix
    
    % Sherman's March
    tic;
    [L1, U1] = shermans(A);
    time_sherman(i) = toc;
    rel_error_sherman(i) = norm(A - L1 * U1, 2) / norm(A, 2);
    
    % Pickett's Charge
    tic;
    [L2, U2] = picketts(A);
    time_pickett(i) = toc;
    rel_error_pickett(i) = norm(A - L2 * U2, 2) / norm(A, 2);
    
    % Crout's Method
    tic;
    [L3, U3] = crouts(A);
    time_crout(i) = toc;
    rel_error_crout(i) = norm(A - L3 * U3, 2) / norm(A, 2);
    
    % Built-in LU
    tic;
    [L4, U4] = lu(A);
    time_builtin(i) = toc;
    rel_error_builtin(i) = norm(A - L4 * U4, 2) / norm(A, 2);
    
    fprintf('Matrix size %d completed\n', n);
end

% Runtime comparison
figure;
hold on;
plot(n_values, time_sherman, 'r', 'DisplayName', 'Sherman');
plot(n_values, time_pickett, 'g', 'DisplayName', 'Pickett');
plot(n_values, time_crout, 'b', 'DisplayName', 'Crout');
plot(n_values, time_builtin, 'k', 'DisplayName', 'Built-in');
title('Runtime Comparison');
xlabel('Matrix Size (n)');
ylabel('Time (seconds)');
legend('show');
grid on;
hold off;

% Relative Error Comparison 
figure;
hold on;

plot(n_values, rel_error_sherman, 'r', 'DisplayName', 'Sherman');
plot(n_values, rel_error_pickett, 'g', 'DisplayName', 'Pickett');
plot(n_values, rel_error_crout, 'b', 'DisplayName', 'Crout');
plot(n_values, rel_error_builtin, 'k', 'DisplayName', 'Built-in');
title('Relative Error Comparison');
xlabel('Matrix Size (n)');
ylabel('Relative Error');
set(gca, 'YScale', 'log'); 
ylim([1e-30 10]);
legend('show');
grid on;
hold off;


fprintf('Sherman relative error for n=%d: %e\n', 23, rel_error_sherman(23));
fprintf('Pickett relative error for n=%d: %e\n', 23, rel_error_pickett(23));
fprintf('Crout relative error for n=%d: %e\n', 23, rel_error_crout(23));
fprintf('builtin relative error for n=%d: %e\n', 23, rel_error_builtin(23));