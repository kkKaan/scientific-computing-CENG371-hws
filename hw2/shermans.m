function [L, U] = shermans(A)
    [m, n] = size(A);
    if m == 1 % Base case
        L = 1;
        U = A;
    else % Recursive 
        [L, U] = shermans(A(1:m-1, 1:n-1));
        L = [L, zeros(m-1, 1); zeros(1, m-1), 1];
        U = [U, A(1:m-1, n); zeros(1, n)];
        for i = 1:m-1
            L(m, i) = A(m, i) / U(i, i);
            U(m, i) = 0;
        end
        U(m, n) = A(m, n) - L(m, 1:m-1) * U(1:m-1, n);
    end
end