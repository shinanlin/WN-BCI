function codewordEnergyDiffMatrix = calcCodewordEnergyDiff(codewordTable, symbolEnergyDiffMatrix)

% �����������Ӧ����֮��������������Ӧ���룩

[tableSize, codeLength] = size(codewordTable);
codewordEnergyDiffMatrix = zeros(tableSize);
codewordTable = codewordTable + 1;

for code_1 = 1:tableSize
    for code_2 = code_1+1:tableSize
        energyDiff = 0;
        for symbol = 1:codeLength
            energyDiff = energyDiff + symbolEnergyDiffMatrix(codewordTable(code_1, symbol), codewordTable(code_2, symbol));
        end
        codewordEnergyDiffMatrix(code_1,code_2) = energyDiff;
        codewordEnergyDiffMatrix(code_2,code_1) = energyDiff;
    end
end

end