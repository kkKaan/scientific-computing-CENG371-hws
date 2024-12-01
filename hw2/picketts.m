function [L, U] = picketts(A)
    [m, n] = size(A);
    
    % Base case
    if m == 1
        L = 1;
        U = A;
        return;
    end
    
    % Recursive case
    [L_sub, U_sub] = picketts(A(2:m, 2:n));
    
    L = eye(m);
    U = zeros(m, n);
    U(1,:) = A(1,:);
    L(2:m,1) = A(2:m,1) / U(1,1);
    L(2:m, 2:m) = L_sub;
    U(2:m, 2:n) = U_sub;
    U(1,n) = A(1,n) - L(2:m,1)' * U(2:m,n);
end