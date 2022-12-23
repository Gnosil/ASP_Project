function para = STDCT(x)

para = {};
for i = 1:length(x)
    data = x{i};
    C=dct(data);
    para = [para C];
end

end