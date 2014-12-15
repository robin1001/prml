close all;
clear all;
%generate data
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



X = data;
K = 3;
%gmm
[mu, sigma_, w] = gmm(X, K);
[maxm index] = max(w, [], 2);
color = ['r', 'g', 'b'];
figure;
for k = 1 : K
    xk = X(find(index == k), :);
    plot(xk(:,1), xk(:,2), [color(k) 'o']);
    hold on;
end

%kmeans
[mu, dis] = kmeans_(X, K);
[mini index] = min(dis, [], 2);
color = ['r', 'g', 'b'];
figure;
for k = 1 : K
    xk = X(find(index == k), :);
    plot(xk(:,1), xk(:,2), [color(k) 'o']);
    hold on;
end
