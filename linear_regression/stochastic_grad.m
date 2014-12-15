x = load('ex2x.dat');
y = load('ex2y.dat');
plot(x, y, 'go');
hold on;

%Ëæ»úÌÝ¶ÈÏÂ½µ
alpha = 0.01;
times = 10000;
x = [x ones(length(x), 1)];
[m, n] = size(x);
w = zeros(n, 1); % init with 1
for i=1:times
    for j=1:m
        delta = 1.0/m * x(j, :)' * (x(j, :) * w - y(j));
        w = w - alpha * delta;
    end
end
plot(x, x * w, 'b-');
disp(w);