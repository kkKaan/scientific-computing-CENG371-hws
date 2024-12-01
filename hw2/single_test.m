% just to test one size for all algorithms

i = 3;
n_values = 1:i;
n = n_values(i);
A = hilb(n); % Generate the Hilbert matrix

% Print the matrix
disp('Matrix A:');
disp(A);

% Sherman's March
tic;
[L1, U1] = shermans(A);
time_sherman = toc;
rel_error_sherman = norm(A - L1 * U1, 2) / norm(A, 2);

% Print L and U
disp('Sherman L:');
disp(L1);
disp('Sherman U:');
disp(U1);
disp('sherman error: ');
disp(rel_error_sherman);

% Pickett's Charge
tic;
[L2, U2] = picketts(A);
time_pickett = toc;
rel_error_pickett = norm(A - L2 * U2, 2) / norm(A, 2);

disp('Pickett L:');
disp(L2);
disp('Pickett U:');
disp(U2);
disp('pickett error: ');
disp(rel_error_pickett);

% Crout's Method
tic;
[L3, U3] = crouts(A);
time_crout = toc;
rel_error_crout = norm(A - L3 * U3, 2) / norm(A, 2);

disp('Crout L:');
disp(L3);
disp('Crout U:');
disp(U3);
disp('crout error: ');
disp(rel_error_crout);

% Built-in LU
tic;
[L4, U4] = lu(A);
time_builtin = toc;
rel_error_builtin = norm(A - L4 * U4, 2) / norm(A, 2);

disp('Built-in L:');
disp(L4);
disp('Built-in U:');
disp(U4);
disp('built-in error: ');
disp(rel_error_builtin);
