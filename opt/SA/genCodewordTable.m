function codewordTable = genCodewordTable(N, M)

% 生成M进制码元，N位码长的码字表

codewordTable = zeros(M^N, N);
for i = 0:M^N-1
    e = log2(i)/log2(M);    % 计算表示第i个码组用到的位数e
    if e == fix(e)
        e = e + 1;
    else
        e = ceil(e);
    end
    f = N - e;              % 不为0的最高位f
    d = i;
    for j = 1:e
        g = floor(d/(M^(e-j)));
        codewordTable(i+1, j+f) = g;
        d = d - g*(M^(e-j)); 
    end
end

end