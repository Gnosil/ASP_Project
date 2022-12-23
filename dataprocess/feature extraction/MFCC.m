function para = MFCC(x,p,winLen,fs)

bank=melbankm(p,winLen,fs,0,0.5,'m');
bank=full(bank);
bank=bank/max(bank(:));

% DCT coefficient,12*p
for k=1:12
  n=0:p-1;
  dctcoef(k,:)=cos((2*n+1)*k*pi/(2*p));
end

w = 1 + 6 * sin(pi*[1:12] ./ 12);
w = w/max(w);


para = {};
for i = 1:length(x)
    data = x{i};
    for n=1:size(data,2)
        y = data(:,n);
        n2=fix(size(y,1)/2)+1;
        t = abs(fft(y));
        t = t.^2;
        c1=dctcoef * log(bank * t(1:n2));
        c2 = c1.*w';
        m(n,:)=c2';
    end

    dtm = zeros(size(m));
    for i=3:size(m,1)-2
      dtm(i,:) = -2*m(i-2,:) - m(i-1,:) + m(i+1,:) + 2*m(i+2,:);
    end
    dtm = dtm / 3;
    ccc = [m dtm];
    ccc = ccc(3:size(m,1)-2,:);

    para = [para ccc];
end
end