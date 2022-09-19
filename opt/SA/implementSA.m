function [reqCodeword, minCodewordEnergyDiff, reqCodewordEnergyDiffMatrix] = implementSA(symbolSignals, codeLength, reqCodewordNum, reheat, fig)

% ģ���˻��㷨�Ż�����
% ���룺��Ԫ�źţ���ά����ͨ����*�źŵ���*��Ԫ���������ֳ��ȣ����������������㷨���ش�����ȱʡ��Ϊ�����أ�,�Ƿ��ͼ
% ������Ż������뼯���뼯��С������Ӧ���룬�뼯�и�����֮�����Ӧ����

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