%perception algorithm, in pattern recognition class
%test data, two class
x = [0 0 1; 0 1 1; 1 0 1; 1 1 1]; %+
flag = [1; 1; -1; -1]; %label of (+ -)
[n, col] = size(x);
for i = 1 : n
    x(i, :) = flag(i) * x(i, :);
end

c = 1;
w = zeros(1, col);
while true
    tag = true; %tag for whether correctly classify or not
    for i = 1 : n
        tmp = w * x(i, :)';
        if tmp <= 0
            w = w + c * x(i, :);
            tag = false;
        end
    end
    if tag == true
        break
    end
end

disp('discriminant function');
disp(w);


