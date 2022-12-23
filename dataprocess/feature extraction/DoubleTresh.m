function para = DoubleTresh(ST_Energy,ST_Zerocorssing,p_E1,p_E2,p_Z,maxsilence,minlen)

para = {};
for i = 1:length(ST_Energy)
    amp = ST_Energy{i};
    zcr = ST_Zerocorssing{i};
    fn = size(amp,1);
    NIS = 4;
    ampth=mean(amp(1:NIS)); 
    zcrth=mean(zcr(1:NIS));
    amp2=p_E2*ampth; amp1=p_E1*ampth;
    zcr2=p_Z*zcrth;
    status  = 0;
    silence = 0;
    count   = 0;
    xn = 1;
    for n = 1:fn
        switch status
            case {0,1}
                if amp(n) > amp1
                    x1(xn) = max(n-count(xn)-1,1);
                    status  = 2;
                    silence(xn) = 0;
                    count(xn)   = count(xn) + 1;
                elseif amp(n) > amp2||zcr(n) > zcr2
                    status = 1;
                    count(xn)  = count(xn) + 1;
                else
                    status  = 0;
                    count(xn)   = 0;
                    x1(xn)=0;
                    x2(xn)=0;
                end
            case 2
                if amp(n) > amp2 && zcr(n) > zcr2
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
                count(xn) = 0;
                silence(xn)=0;
                x1(xn)=0;
                x2(xn)=0;
        end
    end
    
    el=length(x1);
    if x1(el)==0, el=el-1; end 
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