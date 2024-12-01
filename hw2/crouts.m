function [L, U] = crouts(A)
    [m, n] = size(A);
    
    if m == 1
        L = A;
        U = 1;
        return;
    end
    
    [L_sub, U_sub] = crouts(A(2:m, 2:n));
    
    L = zeros(m, m);
    U = eye(m, n);
    
    % First column of L and first row of U
    L(:,1) = A(:,1);
    U(1,2:n) = A(1,2:n) / L(1,1);
    
    % Fill submatrices
    L(2:m, 2:m) = L_sub;
    U(2:m, 2:n) = U_sub;
    
    % Update L's first column properly
    for i = 2:m
        for j = 1:i-1
            L(i,1) = L(i,1) - L(i,j)*U(j,1);
        end
    end
    
    % Update last row of U
    U(m,2:n) = (A(m,2:n) - L(m,1:m-1)*U(1:m-1,2:n));
end