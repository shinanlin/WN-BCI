function [reqCodeword, maxMinDiff, reqCodewordEnergyDiffMatrix] = SAReheat(codeLength, symbolEnergyDiffMatrix, reqCodewordNum, reheat)

% ģ���˻𷽷�
% ���룺�볤����Ԫ��Ӧ�������������������, 
% ���������������С������Ӧ���룬��������Ӧ�����

%% ��ʼ��
symbolNum = size(symbolEnergyDiffMatrix, 1);  % ��Ԫ����
codewordTable = genCodewordTable(codeLength, symbolNum);    % ���ֱ�
codewordEnergyDiffMatrix = calcCodewordEnergyDiff(codewordTable, symbolEnergyDiffMatrix);   % ������Ӧ�������
codewordEnergyDiffMatrix(logical(eye(size(codewordEnergyDiffMatrix)))) = max(max(codewordEnergyDiffMatrix))+1; % ���ڼ�����С������Ӧ����
tableSize = size(codewordTable, 1);

origCodewordSelectIndex = randperm(tableSize);
origCodewordSelectIndex = origCodewordSelectIndex(1:reqCodewordNum); % ��ʼ����ѡ����

% if nargin == 3
%     reheat = 1;
% end

maxCodewordSelectIndex = origCodewordSelectIndex;
maxMinDiff = min(min(codewordEnergyDiffMatrix(maxCodewordSelectIndex, maxCodewordSelectIndex)));


for heatIndex = 1:1+reheat
    heatIndex
    [finalMinDiff, finalCodewordSelectIndex] = SA(maxCodewordSelectIndex, codewordEnergyDiffMatrix, reqCodewordNum);
    if finalMinDiff > maxMinDiff
        maxMinDiff = finalMinDiff;
        maxCodewordSelectIndex = finalCodewordSelectIndex;
    end
end

%% ���
reqCodeword = codewordTable(finalCodewordSelectIndex,:);
reqCodewordEnergyDiffMatrix = codewordEnergyDiffMatrix(finalCodewordSelectIndex, finalCodewordSelectIndex);
reqCodewordEnergyDiffMatrix(logical(eye(size(reqCodewordEnergyDiffMatrix)))) = 0;

end
