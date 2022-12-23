function para = STZ(x)

para = {};
for i = 1:length(x)
    data = x{i};
    ZC = zeros(size(data,2),1);
    for n = 1:size(data,2)
        ZC(n) = sum(([0;data(:,n)].*[data(:,n);0])<0);
    end
    para = [para ZC];
end

end