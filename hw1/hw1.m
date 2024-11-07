function main()
    Q1();
    Q2();
end

function Q1()
    epsilon = 1e-6;
    
    % Initialize arrays for storing f(n) and g(n)
    f_n = zeros(1, 1000);
    g_n = zeros(1, 1000);
    
    % Calculate f(n) and g(n) for each n from 1 to 1000
    for n = 1:1000
        f_n(n) = n * ((n + 1) / n - 1) - 1;
        g_n(n) = f_n(n) / epsilon;
    end
    
    % 1: Plot g(n)
    figure;
    plot(1:1000, g_n, 'b');
    title('Plot of g(n) for n = 1 to 1000');
    xlabel('n');
    ylabel('g(n)');
    grid on;

    % 2: Which values of n satisfy g(n) = 0
    zero_indices = [];
    for n = 1:1000
        if g_n(n) == 0
            zero_indices = [zero_indices, n];  
        end
    end
    disp('Values of n for which g(n) = 0:');
    disp(zero_indices);

    % 3: Explain why majority of g(n) values are non-zero
    non_zero_count = 0;
    for n = 1:1000
        if g_n(n) ~= 0
            non_zero_count = non_zero_count + 1;
        end
    end
    is_majority_non_zero = non_zero_count > 500;
    disp('Is g(n) non-zero for the majority of n?');
    if is_majority_non_zero
        disp('Yes.');
    else
        disp('No.');
    end
end

function Q2()
    disp(' ');
    disp('-----Q2-----');
    disp(' ');

    % Define the array k, matlab shortcut for loops
    n = 1:1e6;
    k = 1 + (1e6 + 1 - n) * 1e-8;
    
    % Convert to single precision for single precision calculations
    k_single = single(k);

    results = struct();
    theoretical_sum = 1e6 + 5000.005;
    
    % Naive Summation
    % a) Double
    tic;
    naive_sum_double = sum(k);
    results.naive_double.time = toc;
    results.naive_double.error = abs(naive_sum_double - theoretical_sum);
    % b) Single
    tic;
    naive_sum_single = sum(k_single);
    results.naive_single.time = toc;
    results.naive_single.error = abs(double(naive_sum_single) - theoretical_sum);
    
    % Compensated (Kahan) Summation
    function s = kahan_sum(x)
        s = 0;
        c = 0;
        for i = 1:length(x)
            y = x(i) - c;
            t = s + y;
            c = (t - s) - y;
            s = t;
        end
    end
    
    % c) Double
    tic;
    kahan_sum_double = kahan_sum(k);
    results.kahan_double.time = toc;
    results.kahan_double.error = abs(kahan_sum_double - theoretical_sum);
    % d) Single
    tic;
    kahan_sum_single = kahan_sum(k_single);
    results.kahan_single.time = toc;
    results.kahan_single.error = abs(double(kahan_sum_single) - theoretical_sum);
    
    % Pairwise Summation
    function s = pairwise_sum(x)
        if length(x) == 1
            s = x;
        else
            mid = floor(length(x) / 2);
            s = pairwise_sum(x(1:mid)) + pairwise_sum(x(mid+1:end));
        end
    end
    
    % e) Double
    tic;
    pairwise_sum_double = pairwise_sum(k);
    results.pairwise_double.time = toc;
    results.pairwise_double.error = abs(pairwise_sum_double - theoretical_sum);
    % f) Single
    tic;
    pairwise_sum_single = pairwise_sum(k_single);
    results.pairwise_single.time = toc;
    results.pairwise_single.error = abs(double(pairwise_sum_single) - theoretical_sum);
    
    fprintf('Method            | Precision | Sum                     | Error             | Runtime (s)\n');
    fprintf('------------------|-----------|-------------------------|-------------------|-------------\n');
    fprintf('Naive             | Double    | %.15f | %.15f | %.5f\n', naive_sum_double, results.naive_double.error, results.naive_double.time);
    fprintf('Naive             | Single    | %.15f | %.15f | %.5f\n', naive_sum_single, results.naive_single.error, results.naive_single.time);
    fprintf('Kahan             | Double    | %.15f | %.15f | %.5f\n', kahan_sum_double, results.kahan_double.error, results.kahan_double.time);
    fprintf('Kahan             | Single    | %.15f | %.15f | %.5f\n', kahan_sum_single, results.kahan_single.error, results.kahan_single.time);
    fprintf('Pairwise          | Double    | %.15f | %.15f | %.5f\n', pairwise_sum_double, results.pairwise_double.error, results.pairwise_double.time);
    fprintf('Pairwise          | Single    | %.15f | %.15f | %.5f\n', pairwise_sum_single, results.pairwise_single.error, results.pairwise_single.time);
end