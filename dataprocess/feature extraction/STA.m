function para = STA(x)

para = {};
for i = 1:length(x)
    data = x{i};
    A = zeros(size(data));
    for n = 1:size(data,2)
        R = xcorr(data(:,n));
        A(:,n) = R(size(data,1):end);
    end
    para = [para A];
end

end