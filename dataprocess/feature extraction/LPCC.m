function para = LPCC(x,n_lpc,n_lpcc)  
    para = {};
    for i = 1:length(x)
        data = x{i};
        LPcc = [];
        for m = 2:size(data,1)
            y = data(m,:);
            lpcc=zeros(n_lpcc,1);
            lpcc(1)=y(1);
            for n=2:n_lpc                                     % 计算n=2,...,p的lpcc
                lpcc(n)=y(n);
                for l=1:n-1
                    lpcc(n)=lpcc(n)+y(l)*lpcc(n-l)*(n-l)/n;
                end
            end
            for n=n_lpc+1:n_lpcc                              % 计算n>p的lpcc
                lpcc(n)=0;
                for l=1:n_lpc
                    lpcc(n)=lpcc(n)+y(l)*lpcc(n-l)*(n-l)/n;
                end
            end
            lpcc=-lpcc;
            LPcc = [LPcc lpcc];
        end
        para = [para LPcc];
    end
end