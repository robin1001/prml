x = load('ex2x.dat');
y = load('ex2y.dat');
plot(x, y, 'go');
hold on;

%最小二乘
x = [x ones(length(x), 1)];
w = inv(x'* x) * x' * y;
plot(x, x * w, 'r-');
hold on;
disp(w);

%梯度下降
alpha = 0.05;
times = 1000;
[m, n] = size(x);
w = zeros(n, 1); % init with 1
for i=1:times
    delta = 1.0/m * x' * (x*w - y);
    w = w - alpha * delta;
end
plot(x, x * w, 'b-');
disp(w);