function para = StFt(x)
    para = {};
    for i = 1:length(x)
        data = x{i};
        C=fft(data);
        winLen = size(data,1);
        hWL = floor(winLen/2);
        % consider the phase difference by window
        C = C(1:hWL+1,:);
        para = [para C];
    end
end