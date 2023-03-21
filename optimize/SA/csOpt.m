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

save('CodespaceSEQ.mat')

seq = load('codeSpace.mat');

% picked.S10 = seq.S10;

picked = seq;

codeLength = 1;
reqCodewordNum = 40;
reheat = 3;
srate = 250;
latency = 0.14*srate;
N = 10000;
winLENs = 0.3;

name = picked.name;

for j=1:length(winLENs)

    win = winLENs(j)*srate;

    [opt(j).simulate.code, opt(j).simulate.distance, opt(j).simulate.mat] = implementSA(picked.simulate(1,latency:latency+win,:), codeLength, reqCodewordNum, reheat,'n');
    
    opt(j).winLEN = winLENs(j);
    opt(j).N = N;

end

eval([name,'=opt'])
save('CodespaceSEQ.mat',name,'-append')
