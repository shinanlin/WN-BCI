function selectCodewordIndex = perturbSelectCodeword(selectCodewordIndex, codewordSelectNum, codeSetSize)

% 扰动所选码字,每次扰动一个

codewordNewSelect = Ranint(1, codeSetSize); % 用于替换的码字
codewordPerturb = Ranint(1, codewordSelectNum);    % 被替换的码字

while ismember(codewordNewSelect, selectCodewordIndex)
    codewordNewSelect = Ranint(1,codeSetSize);    % 避免出现码字重复
end

selectCodewordIndex(codewordPerturb) = codewordNewSelect;

end