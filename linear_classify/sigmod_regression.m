data = load('data.txt');

class1 = data((find(data(:, 3) == 1)), :);
class2 = data((find(data(:, 3) == 0)), :);
t = data(:, 3);

w = ones(3, 1);
x = [data(:, 1:2) ones(num, 1)];
epoch = 1000;
alpha = 0.001;
sigmod = @(x) (1.0./(1+exp(-x)));
for i = 1:epoch
   y = sigmod(x * w);
   dw = x' * (t - y);
   w = w + alpha * dw;
end

plot(class1(:, 1), class1(:, 2), 'ro');
hold on;
plot(class2(:, 1), class2(:, 2), 'bo');
hold on;
y = -(w(3) + w(1) * x(:, 1)) / w(2);
plot(data(:, 1), y, 'g-');
