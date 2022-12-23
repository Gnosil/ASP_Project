clc; clear all; close all;

folder = '.\Audio';
data = load_data(folder);
wave_fs = 16000;

data_ex{1} = data{1}(1320000:1480000);

%% preprocessing
addpath('./preprocess')
winLen = 1600; skip = winLen/4;

% pre-emphasis
data_ex{1} = filter([1,-0.95],1,data_ex{1});

% remove baseline drift
n = 1;
data_ex{1} = baseline(data_ex{1}, wave_fs, 1);

% frame
[data_frame,data_fram_nowin,Ls,Bounds] = enframe(data_ex,winLen,skip,wave_fs);



%% Feature Extraction
addpath('./feature extraction/')

% Time based
% short time energy (not average)
ST_Energy = STE(data_fram_nowin);
% short time magnitude (not average)
ST_Magnitude = STM(data_fram_nowin);
% short time zero crossing rate
ST_Zerocorssing = STZ(data_fram_nowin);
% short time autocorrelation
ST_Autocorrelation = STA(data_fram_nowin);
% short time magnitude difference
ST_MagDif = STMD(data_fram_nowin);

% Freq based
% short time fourier transform
ST_FT = StFt(data_frame);
% imagesc(20*log10(abs(ST_FT{1})+eps));
% title('STFT');
% colormap(jet)
% short time cepstrum
ST_CP = STCP(data_frame);
% short time DCT
ST_DCT = STDCT(data_frame);

% MFCC based
p_MFCC = 12;
MFCC_data = MFCC(data_frame,p_MFCC,winLen,wave_fs);

% LPC based
% LPC
p_LPC = 12;
LPC_data = LPC(data_frame,p_LPC);
% LPCC
p_LPCC = 12;
LPCC_data = LPCC(LPC_data,p_LPC,p_LPCC);
% LSF ??

% simple VAD
% double threshold
p_E1 = 2; p_E2 = 4;p_Z = 2;
maxsilence = 15;
minlen  = 5; 
NIS = 8;
DT = DoubleTresh(ST_Energy,ST_Zerocorssing,p_E1,p_E2,p_Z,maxsilence,minlen);
% Spectral Entropy
th1=0.99;
th2=0.96;
maxsilence = 8;
minlen  = 5; 
NIS = 8;
SER = SE_Ratio(ST_FT,th1,th2,maxsilence,minlen,wave_fs,winLen,NIS);
% Zero Energy Ratio
a=2; b=1;NIS=8;
ZER = ZE_Ratio(ST_Energy,ST_Zerocorssing,a,b,NIS);
% Energy Entropy Ratio
a = 2;th1 = 0.05;th2 = 0.1;
EER = EE_Ratio(ST_FT,a,th1,th2,NIS);
% Log-spectral distance
NIS = 8;
LSD = LS_Dist(ST_FT,NIS);
