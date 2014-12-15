x1 = randn(1, 300);
y1 = randn(1, 300);
plot(x1, y1, 'ro');
hold on

x2 = randn(1, 100) + 4;
y2 = randn(1, 100) - 2;
plot(x2, y2, 'go');
hold on

x3 = randn(1, 100) + 4;
y3 = randn(1, 100) + 4;
plot(x3, y3, 'bo');

x = [x1 x2 x3];
y = [y1 y2 y3];

data = [x' y'];

save('data.mat', 'data')