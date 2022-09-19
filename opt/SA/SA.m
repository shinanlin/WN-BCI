function [finalMinDiff, finalCodewordSelectIndex] = SA(origCodewordSelectIndex, codewordEnergyDiffMatrix, reqCodewordNum)

% ģ���˻��㷨����
% ���룺��ʼ����ѡ��������Ӧ�������������������
% ������Ż�����ѡ����С������Ӧ����

%% ��ʼ��
tableSize = size(codewordEnergyDiffMatrix, 1);

temperature = 10000 * reqCodewordNum;                                       % ��ʼ�¶�
iterTime = 50 * reqCodewordNum;                                             % ��������

finalCodewordSelectIndex = origCodewordSelectIndex;                         % �������ս�
finalMinDiff = min(min(codewordEnergyDiffMatrix(origCodewordSelectIndex, origCodewordSelectIndex)));

bestCodewordSelectIndex = origCodewordSelectIndex;                          % �������Ž�
bestMinDiff = finalMinDiff;

while temperature > 0.001
    for iter = 1:iterTime
        preMinDiff = min(min(codewordEnergyDiffMatrix(finalCodewordSelectIndex,finalCodewordSelectIndex)));
        curCodewordSelectIndex = perturbSelectCodeword(finalCodewordSelectIndex, reqCodewordNum, tableSize);
        curMinDiff = min(min(codewordEnergyDiffMatrix(curCodewordSelectIndex,curCodewordSelectIndex)));
        
        MD_diff = preMinDiff - curMinDiff;                                  % Metropolis׼��
        if MD_diff < 0 || exp(-MD_diff/temperature) > rand()
            finalCodewordSelectIndex = curCodewordSelectIndex;
            finalMinDiff = curMinDiff;
        end
        
        if curMinDiff > bestMinDiff                                         % �������Ž�
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
