%lmse algorithm

%example 1, linear separable
x = [0 0 1; 0 1 1; 1 0 1; 1 1 1]; %+
flag = [1; 1; -1; -1]; %label of (+ -)
n = size(x, 1);
for i = 1 : n
    x(i, :) = flag(i) * x(i, :);
end
w = inv(x'x) * x' * b



%example 2, linear inseparable
x = [0 0 1; 0 1 1; 1 0 1; 1 1 1]; %+
flag = [1; -1; -1; 1]; %label of (+ -)
n = size(x, 1);
for i = 1 : n
    x(i, :) = flag(i) * x(i, :);
end
