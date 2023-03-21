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

t = 240*1;
% load('Example symbolSignals.mat');
load('response.mat')
codeLength = 1;
winLEN = 1;
srate = 240;
reqCodewordNum = 40;
reheat = 2;

N = 1e4;
Ns = 100:1000:N;
latency = 0*srate;

for i = 1:size(Ns,2)
    
    i

    n = Ns(i);
    
    win = winLEN*srate;

    [code, distance, mat] = implementSA(simuLarge(1,1:win,1:n), codeLength, reqCodewordNum, reheat,'n');
    
    opt(i).code = code;
    opt(i).distance = distance;
    opt(i).mat = mat;
 
    opt(i).winLEN = winLEN;
    opt(i).N = n;

    save('search.mat','opt')

end
