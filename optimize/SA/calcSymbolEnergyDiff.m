function symbolEnergyDiffMatrix = calcSymbolEnergyDiff(symbolSignals)

% 计算码元响应信号两两之间的能量差（码元响应距离）
% 输入：码元信号（多维矩阵形式:通道数*信号点数*码元数量）
% 输出：码元响应距离矩阵

[channelNum, ~, symbolNum] = size(symbolSignals);
symbolEnergyDiffMatrix = zeros(symbolNum);

for symbol_1 = 1:symbolNum
    for symbol_2 = symbol_1+1:symbolNum
        % 分别求各通道EEG信号的能量差，取均值
        symbolEnergyDiffMatrix(symbol_1, symbol_2) = sum(sum((symbolSignals(:, :, symbol_1)-symbolSignals(:, :, symbol_2)).^2))/channelNum;
        symbolEnergyDiffMatrix(symbol_2, symbol_1) = symbolEnergyDiffMatrix(symbol_1, symbol_2);
    end
end

end