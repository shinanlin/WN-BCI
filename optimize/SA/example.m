% ʾ������
% ���� implementSA ������������Ż���ı��뷽��
% ���룺
    % symbolSignals     ��Ԫ�źţ���ά����ͨ����*�źŵ���*��Ԫ����
    % codeLength        ���ֳ���
    % reqCodewordNum    ������������
    % reheat            �㷨���ش�����ȱʡʱĬ��Ϊ0�������ؿ��ܵõ����ý�����������ĸ���ʱ��
    % fig               �Ƿ����������Ӧ����ͼ��ȱʡʱĬ��Ϊ'y'��ȡ��'n'��
% �����
    % reqCodeword                   �Ż������뼯
    % minCodewordEnergyDiff         ��С������Ӧ����
    % reqCodewordEnergyDiffMatrix   ������Ӧ����
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
