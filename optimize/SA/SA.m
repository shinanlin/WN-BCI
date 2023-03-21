function [finalMinDiff, finalCodewordSelectIndex] = SA(origCodewordSelectIndex, codewordEnergyDiffMatrix, reqCodewordNum)

% 模拟退火算法主体
% 输入：初始码字选择，码字响应距离矩阵，所需码字数量
% 输出：优化码字选择，最小码字响应距离

%% 初始化
tableSize = size(codewordEnergyDiffMatrix, 1);

temperature = 10000 * reqCodewordNum;                                       % 初始温度
iterTime = 50 * reqCodewordNum;                                             % 迭代次数

finalCodewordSelectIndex = origCodewordSelectIndex;                         % 迭代最终解
finalMinDiff = min(min(codewordEnergyDiffMatrix(origCodewordSelectIndex, origCodewordSelectIndex)));

bestCodewordSelectIndex = origCodewordSelectIndex;                          % 迭代最优解
bestMinDiff = finalMinDiff;

while temperature > 0.001
    for iter = 1:iterTime
        preMinDiff = min(min(codewordEnergyDiffMatrix(finalCodewordSelectIndex,finalCodewordSelectIndex)));
        curCodewordSelectIndex = perturbSelectCodeword(finalCodewordSelectIndex, reqCodewordNum, tableSize);
        curMinDiff = min(min(codewordEnergyDiffMatrix(curCodewordSelectIndex,curCodewordSelectIndex)));
        
        MD_diff = preMinDiff - curMinDiff;                                  % Metropolis准则
        if MD_diff < 0 || exp(-MD_diff/temperature) > rand()
            finalCodewordSelectIndex = curCodewordSelectIndex;
            finalMinDiff = curMinDiff;
        end
        
        if curMinDiff > bestMinDiff                                         % 保存最优解
            bestMinDiff = curMinDiff; 
            
%             bestMinDiff
            bestCodewordSelectIndex = curCodewordSelectIndex;
        end
    end
    % finalCodewordSelectIndex = sort(finalCodewordSelectIndex);
    temperature = temperature * 0.99;
end

if bestMinDiff > finalMinDiff
    finalCodewordSelectIndex = bestCodewordSelectIndex;
end
finalCodewordSelectIndex = sort(finalCodewordSelectIndex);

end
