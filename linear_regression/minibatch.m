x = load('ex2x.dat');
y = load('ex2y.dat');
plot(x, y, 'go');
hold on;

%Ëæ»úÌÝ¶ÈÏÂ½µ
alpha = 0.01;
times = 5000;
x = [x ones(length(x), 1)];
batch = 10;
[m, n] = size(x);
w = zeros(n, 1); % init with 1
for i=1:times
    for j=1:batch:m-1
        delta = 1.0/m * x(j:j+batch-1, :)' * (x(j:j+batch-1, :) * w - y(j:j+batch-1));
        w = w - alpha * delta;
    end
end
plot(x, x * w, 'b-');
disp(w);