function para = STCP(x)

para = {};
for i = 1:length(x)
    data = x{i};
    fftxabs = abs(fft(data));
    xhat = real(ifft(log(fftxabs)));
    para = [para xhat];
end
end
