function [processed_wave,processed_wave_nowin,Ls,Bounds] = enframe(data_ex,winLen,skip,wave_fs)

win_hann = win_norm(hann(winLen,'symmetric'), skip);
processed_wave = {};
Ls = {};
Bounds = {};
processed_wave_nowin = {};

for i = 1:length(data_ex)
    data = data_ex{i};
    data=data-mean(data);                        
    data=data/max(abs(data));
    fs = wave_fs;
    ls = ceil((length(data)+2*(winLen-skip)-winLen)/skip)*skip+winLen;
    bounds = [winLen-skip,length(data)+(winLen-skip)];
    data = [zeros(winLen-skip,1);data;zeros(ls-length(data)-2*(winLen-skip),1);zeros(winLen-skip,1)];
    idx = (1:winLen)' + (0:skip:ls-winLen-skip);
    sliced = data(idx).*win_hann;
    sliced_nowin = data(idx);
    processed_wave = [processed_wave sliced];
    processed_wave_nowin = [processed_wave_nowin sliced_nowin];
    Ls = [Ls ls];
    Bounds = [Bounds bounds];
end

end
