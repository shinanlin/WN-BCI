function [reqCodeword, maxMinDiff, reqCodewordEnergyDiffMatrix] = SAReheat(codeLength, symbolEnergyDiffMatrix, reqCodewordNum, reheat)

% 模拟退火方法
% 输入：码长，码元响应距离矩阵，所需码字数量, 
% 输出：编码结果，最小码字响应距离，各码字响应距离表

%% 初始化
symbolNum = size(symbolEnergyDiffMatrix, 1);  % 码元数量
codewordTable = genCodewordTable(codeLength, symbolNum);    % 码字表
codewordEnergyDiffMatrix = calcCodewordEnergyDiff(codewordTable, symbolEnergyDiffMatrix);   % 码字响应距离矩阵
codewordEnergyDiffMatrix(logical(eye(size(codewordEnergyDiffMatrix)))) = max(max(codewordEnergyDiffMatrix))+1; % 用于计算最小码字响应距离
tableSize = size(codewordTable, 1);

origCodewordSelectIndex = randperm(tableSize);
origCodewordSelectIndex = origCodewordSelectIndex(1:reqCodewordNum); % 初始化所选码字

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

%% 结果
reqCodeword = codewordTable(finalCodewordSelectIndex,:);
reqCodewordEnergyDiffMatrix = codewordEnergyDiffMatrix(finalCodewordSelectIndex, finalCodewordSelectIndex);
reqCodewordEnergyDiffMatrix(logical(eye(size(reqCodewordEnergyDiffMatrix)))) = 0;

end
