function para = SE_Ratio(ST_FT,th1,th2,maxsilence,minlen,wave_fs,winLen,NIS)
para = {};
for i = 1:length(ST_FT)
    data = ST_FT{i};
    fn=size(data,2);
    df=wave_fs/winLen; 
    fx1=fix(250/df)+1; fx2=fix(3500/df)+1;
    km=floor(winLen/8);K=0.5;
    for i=1:fn
        A=abs(data(:,i));
        E=zeros(winLen/2+1,1);            
        E(fx1+1:fx2-1)=A(fx1+1:fx2-1); 
        E=E.*E; 
        P1=E/sum(E);
        index=find(P1>=0.9); 
        if ~isempty(index), E(index)=0; end 
        for m=1:km
            Eb(m)=sum(E(4*m-3:4*m));
        end
        prob=(Eb+K)/sum(Eb+K);             
        Hb(i) = -sum(prob.*log(prob+eps)); 
    end   
    Enm=multimidfilter(Hb,10);
    Enm = log(m+eps)-Enm;
    Me=min(Enm);  
    eth=mean(Enm(1:NIS));
    Det=eth-Me;
    T1=th1*Det+Me;
    T2=th2*Det+Me;
    
    
    status  = 0;
    count   = 0;
    silence = 0;
    xn=1;
    for n=2:fn
       switch status
       case {0,1}                         
          if Enm(n) < T2               
             x1(xn) = max(n-count(xn)-1,1);
             status  = 2;
             silence(xn) = 0;
             count(xn)   = count(xn) + 1;
          elseif Enm(n) < T1            
             status = 1;
             count(xn)  = count(xn) + 1;
          else                        
             status  = 0;
             count(xn)   = 0;
             x1(xn)=0;
             x2(xn)=0;
          end
       case 2
          if Enm(n) < T1              
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

    para = [para voiceseg];
end

end