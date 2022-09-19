function symbolEnergyDiffMatrix = calcSymbolEnergyDiff(symbolSignals)

% ������Ԫ��Ӧ�ź�����֮����������Ԫ��Ӧ���룩
% ���룺��Ԫ�źţ���ά������ʽ:ͨ����*�źŵ���*��Ԫ������
% �������Ԫ��Ӧ�������

[channelNum, ~, symbolNum] = size(symbolSignals);
symbolEnergyDiffMatrix = zeros(symbolNum);

for symbol_1 = 1:symbolNum
    for symbol_2 = symbol_1+1:symbolNum
        % �ֱ����ͨ��EEG�źŵ������ȡ��ֵ
        symbolEnergyDiffMatrix(symbol_1, symbol_2) = sum(sum((symbolSignals(:, :, symbol_1)-symbolSignals(:, :, symbol_2)).^2))/channelNum;
        symbolEnergyDiffMatrix(symbol_2, symbol_1) = symbolEnergyDiffMatrix(symbol_1, symbol_2);
    end
end

end