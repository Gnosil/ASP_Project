function para = LS_Dist(ST_FT,NIS)

para = {};
for i = 1:length(ST_FT)

    Y = ST_FT{i};
    Y = abs(Y);
    N = mean(Y(:,end-1:end),2);
    
    SpectralDist = 10*(log10(Y)-log10(N));
    SpectralDist(find(SpectralDist<0))=0;
    result = mean(SpectralDist);
    para= [para result];
end

end