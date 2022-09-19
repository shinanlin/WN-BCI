% 示例程序
% 调用 implementSA 函数即可输出优化后的编码方案
% 输入：
    % symbolSignals     码元信号（多维矩阵：通道数*信号点数*码元数）
    % codeLength        码字长度
    % reqCodewordNum    所需码字数量
    % reheat            算法重载次数（缺省时默认为0），重载可能得到更好结果，但会消耗更多时间
    % fig               是否绘制码字响应距离图（缺省时默认为'y'，取消'n'）
% 输出：
    % reqCodeword                   优化编码码集
    % minCodewordEnergyDiff         最小码字响应距离
    % reqCodewordEnergyDiffMatrix   码字响应距离
% 

close all;
clear;
clc;

save('optimized.mat')

seq = load('forOptimized.mat');

codeLength = 1;
reqCodewordNum = 40;
srate = 240;
latency = 0.14*srate;
N = 160;
winLENs = 0.3:0.1:0.5;
reheats = 0:5:10;
subNames = fieldnames(seq);

for i  = 1:length(subNames)
    
    i

    name = subNames(i);
    sub = seq.(name{1});    
    for k = 1:length(reheats)
        reheat =  reheats(k);
        for j=1:length(winLENs)
            
            win = winLENs(j)*srate;
            inx = (j-1)*length(winLENs)+k;
            

            [opt(inx).real.code, opt(inx).real.distance, opt(inx).real.mat] = implementSA(sub.pattern(1,latency:latency+win,:), codeLength, reqCodewordNum, reheat,'n');
            [opt(inx).simulate.code, opt(inx).simulate.distance, opt(inx).simulate.mat] = implementSA(sub.simulate(1,latency:latency+win,:), codeLength, reqCodewordNum, reheat,'n');
            
            opt(inx).winLEN = winLENs(j);
            opt(inx).N = N;
            opt(inx).reheat = reheat;
     
        end
    end

    eval([sub.name,'=opt'])
    save('optimized.mat',sub.name,'-append')
end

    