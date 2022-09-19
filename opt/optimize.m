rngNumber = 253;
rng(rngNumber)
refreshRate = 60;
winLEN = 1;

sampleSize = refreshRate*winLEN;
poolSize = 1e6;
pool = rand(poolSize,sampleSize);
pickSize = 160;

pickNUM = 1e6;

values = zeros(pickNUM,1);
pickLEN = refreshRate*0.5;

for i=1:pickNUM
    
    rng(i)
    picks = pool(randi(poolSize,1,pickSize),1:pickLEN);

    p = corr(picks);
    p = p-diag(diag(p));
    values(i,:) = sum(p,'all');
    
end

miminum = min(values);
index = find(values==miminum);
rng(index)
optimal = pool(randi(poolSize,1,pickSize),:);

save('optimal.mat','index','optimal','rngNumber','miminum')