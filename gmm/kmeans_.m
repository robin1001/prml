function [mu, dis] = kmeans_(X, K)
    [N, D] = size(X);
    %init...
    mu = X(randperm(N, K), :);
    dis = zeros(N, K);
    pre = inf;
    while true        
        %M
        for i = 1 : N
            for k = 1 : K
                dis(i, k) = sum((X(i, :) - mu(k, :)).^2);
            end
        end
        [mini index] = min(dis, [], 2);
        
        jfunc = 0;
        for k = 1 : K
            cur = find(index == k);
            jfunc = jfunc + sum(mini(cur));
            mu(k, :) = sum(X(cur, :), 1) / length(cur); 
        end
        
        if pre - jfunc < 1e-6
            break;
        end
        pre = jfunc;
    end
end