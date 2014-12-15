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

