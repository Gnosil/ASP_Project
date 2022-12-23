function para = STM(x)

para = {};
for i = 1:length(x)
    data = x{i};
    Magnitude = zeros(size(data,2),1);
    for n = 1:size(data,2)
        Magnitude(n) = sum(data(:,n));
    end
    para = [para Magnitude];
end

end