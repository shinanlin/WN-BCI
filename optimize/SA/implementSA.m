function [reqCodeword, minCodewordEnergyDiff, reqCodewordEnergyDiffMatrix] = implementSA(symbolSignals, codeLength, reqCodewordNum, reheat, fig)

% 模拟退火算法优化编码
% 输入：码元信号（多维矩阵：通道数*信号点数*码元数），码字长度，所需码字数量，算法重载次数（缺省即为不重载）,是否绘图
% 输出：优化编码码集，码集最小码字响应距离，码集中各码字之间的响应距离

symbolEnergyDiffMatrix = calcSymbolEnergyDiff(symbolSignals);

if nargin == 3
    reheat = 0;
    fig = 'y';
end

if nargin == 4
    fig = 'y';
end

[reqCodeword, minCodewordEnergyDiff, reqCodewordEnergyDiffMatrix] = SAReheat(codeLength, symbolEnergyDiffMatrix, reqCodewordNum, reheat);

%%
if fig == 'y'
    figure;
    fontSize = 10;
    fontName = 'Arial';%'Times New Roman';
    h = heatmap(reqCodewordEnergyDiffMatrix);
    h.title('Codeword Energy Differences');
    h.FontName = fontName;
    h.FontSize = fontSize;
    h.ColorbarVisible = 'on';
end

end