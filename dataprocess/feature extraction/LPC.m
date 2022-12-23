function para = LPC(x,p)
    para = {};
    for i = 1:length(x)
        data = x{i};
        for n = 1:size(data,2)
            y = data(:,n);
            m(n,:)=lpc(y,p);
        end
        para = [para m];
    end
end