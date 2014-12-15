function W = calc_w(p, x, k)
%x, y训练数据
%p 预测的点
   [m, n] = size(x);
    W = zeros(m, m);
    for i=1:m
        u = x(i, :) - p;
        W(i, i) = exp(u * u'/ (-2*k*k));
    end

%vectorize repeat mat
%     u = x - repmat(p, [size(x, 1), 1]); 
%     w = exp(sum(u .* u, 2)/ (-2*k*k));
%     W = diag(w);   
end