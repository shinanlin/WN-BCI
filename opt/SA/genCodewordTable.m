function codewordTable = genCodewordTable(N, M)

% ����M������Ԫ��Nλ�볤�����ֱ�

codewordTable = zeros(M^N, N);
for i = 0:M^N-1
    e = log2(i)/log2(M);    % �����ʾ��i�������õ���λ��e
    if e == fix(e)
        e = e + 1;
    else
        e = ceil(e);
    end
    f = N - e;              % ��Ϊ0�����λf
    d = i;
    for j = 1:e
        g = floor(d/(M^(e-j)));
        codewordTable(i+1, j+f) = g;
        d = d - g*(M^(e-j)); 
    end
end

end