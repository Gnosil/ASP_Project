function para = STMD(x)

para = {};
for i = 1:length(x)
    data = x{i};
    A = zeros(size(data));
    for n = 1:size(data,2)
        wave = data(:,n);
        for m = 1:size(data,1)
            A(m,n) = mean(abs(wave(m:end)-wave(1:end-m+1))); 
        end
    end
    para = [para A];
end

end