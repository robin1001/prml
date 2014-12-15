function [mu, sigma_, w] = gmm(X, K)
    %load('data.mat');
    %X = data;
    %K = 3;
    [N, D] = size(X);
    %init...
    mu = X(randperm(N, K), :);
    sigma_ = zeros(D, D, K);
    w = zeros(N, K);
    px = zeros(N, K);
    pk = zeros(1,K); 
    %assign (x - mu)^2
    dis = zeros(N, K);
    for i = 1 : N
        for j = 1 : K
            dis(i, j) = sum((X(i, :) - mu(j, :)).^2);
        end
    end
    [mini index] = min(dis, [], 2);
    for k = 1 : K
        xk = X(find(index == k), :);
        pk(k) = size(xk, 1) / N;
        sigma_(:, :, k) = cov(xk);
    end
    
    pre = -inf;
    while true
        % E step, estimate w(i,k)
        for k = 1 : K
            a = 1 / ((2*pi)^(D/2) * sqrt(det(sigma_(:, :, k))));
            Y = (X - repmat(mu(k, :), N, 1));
            b = -0.5 * sum(Y * inv(sigma_(:, :, k)) .* Y, 2);
            px(:, k) = a * exp(b);
        end
        for i = 1 : N
            all = sum(px(i, :) .* pk);
            for k = 1 : K
                w(i, k) = pk(k) * px(i, k) / all;
            end
        end
        % M step, update pk, mu, sigma
        sk = sum(w, 1);
        pk = sk / N;
        mu = diag(1./sk) * w' * X;
        for k = 1 : K
            Y = (X - repmat(mu(k, :), N, 1));
            sigma_(:, :, k) = (Y' * diag(w(:, k)) * Y) / sk(k);
        end
        
        %convergence
        jfunc = sum(log(px * pk'));
        if jfunc - pre < 1e-6
            break;
        end
        pre = jfunc;
    end
end
