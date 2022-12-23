function para = EE_Ratio(ST_FT,a,th1,th2,NIS)

para = {};
for i = 1:length(ST_FT)
    Sp = ST_FT{i};
    Sp = abs(Sp);
    Energy = sum(Sp.*Sp);
    Esum = log10(1+Energy/a);
    prob = Sp./sum(Sp);
    H = -sum(prob.*log(prob+eps));
    Ef = sqrt(1 + abs(Esum./H)); 
    Epara=multimidfilter(Ef,10); 
    Me = max(Epara);
    eth=mean(Epara(1:NIS));
    Det=Me-eth;
    T1=th1*Det+eth;
    T2=th2*Det+eth;
    dst1 = Epara;
    fn=size(dst1,2);
    maxsilence = 8; 
    minlen  = 5;    
    status  = 0;
    count   = 0;
    silence = 0;
    
    xn=1;
    for n=2:fn
       switch status
       case {0,1} 
          if dst1(n) > T2
             x1(xn) = max(n-count(xn)-1,1);
             status  = 2;
             silence(xn) = 0;
             count(xn)   = count(xn) + 1;
          elseif dst1(n) > T1
             status = 1;
             count(xn)  = count(xn) + 1;
          else 
             status  = 0;
             count(xn)   = 0;
             x1(xn)=0;
             x2(xn)=0;
          end
       case 2
          if dst1(n) > T1 
             count(xn) = count(xn) + 1;
          else
             silence(xn) = silence(xn)+1;
             if silence(xn) < maxsilence
                count(xn)  = count(xn) + 1;
             elseif count(xn) < minlen
                status  = 0;
                silence(xn) = 0;
                count(xn)   = 0;
             else
                status  = 3;
                x2(xn)=x1(xn)+count(xn);
             end
          end
       case 3
            status  = 0;          
            xn=xn+1; 
            count(xn)   = 0;
            silence(xn)=0;
            x1(xn)=0;
            x2(xn)=0;
       end
    end   
    el=length(x1);
    if x1(el)==0, el=el-1; end 
    if el==0, return; end
    if x2(el)==0 
        fprintf('Error: Not find endding point!\n');
        x2(el)=fn;
    end
    SF=zeros(1,fn);
    NF=ones(1,fn);
    for i=1 : el
        SF(x1(i):x2(i))=1;
        NF(x1(i):x2(i))=0;
    end
    speechIndex=find(SF==1); 
    voiceseg=findSegment(speechIndex);
    para= [para voiceseg];
end


end