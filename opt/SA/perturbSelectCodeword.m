function selectCodewordIndex = perturbSelectCodeword(selectCodewordIndex, codewordSelectNum, codeSetSize)

% �Ŷ���ѡ����,ÿ���Ŷ�һ��

codewordNewSelect = Ranint(1, codeSetSize); % �����滻������
codewordPerturb = Ranint(1, codewordSelectNum);    % ���滻������

while ismember(codewordNewSelect, selectCodewordIndex)
    codewordNewSelect = Ranint(1,codeSetSize);    % ������������ظ�
end

selectCodewordIndex(codewordPerturb) = codewordNewSelect;

end