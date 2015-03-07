x = load('ex2x.dat');
y = load('ex2y.dat');
plot(x, y, 'go');
hold on;

x2 = [x ones(length(x), 1)];
[m, n] = size(x);
y2 = zeros(m, 1);

k = 0.1;
for i=1:m
    W = calc_w(x2(i, :), x2, k);
    w = inv(x2'* W * x2) * x2'* W * y;
    y2(i) = x2(i, :) * w;
end
plot(x, y2, 'r-');

k=1;
for i=1:m
    W = calc_w(x2(i, :), x2, k);
    w = inv(x2'* W * x2) * x2'* W * y;
    y2(i) = x2(i, :) * w;
end
plot(x, y2, 'b-');

k=10;
for i=1:m
    W = calc_w(x2(i, :), x2, k);
    w = inv(x2'* W * x2) * x2'* W * y;
    y2(i) = x2(i, :) * w;
end
plot(x, y2, 'g-');



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