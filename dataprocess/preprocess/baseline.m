function y = baseline(x, fs, m)

x=x(:);
N=length(x);
t= (0: N-1)'/fs;
a=polyfit(t, x, m);
xtrend=polyval(a, t);
y=x-xtrend;

end