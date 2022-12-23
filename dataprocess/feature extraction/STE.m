function para = STE(x)

para = {};
for i = 1:length(x)
    data = x{i};
    Energy = zeros(size(data,2),1);
    for n = 1:size(data,2)
        Energy(n) = data(:,n)'*data(:,n);
    end
    para = [para Energy];
end

end