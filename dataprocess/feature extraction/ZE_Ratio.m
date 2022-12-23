function para = ZE_Ratio(ST_Energy,ST_Zerocorssing,a,b,NIS)

para = {};
for i = 1:length(ST_Energy)
    energy = ST_Energy{i};
    temp1=log10(1+energy/a);
    ZC = ST_Zerocorssing{i};
    Ecr=temp1./(ZC+b);
    Epara=multimidfilter(Ecr,10);
    dth=mean(Epara(1:NIS));
    T1=1.2*dth;T2=2*dth;
    dst1 = Epara;
    fn=size(dst1,1);
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