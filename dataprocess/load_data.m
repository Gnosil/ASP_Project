function [data fs] = load_data(folder)

filelist = dir([folder '\*.wav']);
data = {};
fs = zeros(1, length(filelist));

for file = 1:length(filelist)
    [wave_data, wave_fs] = audioread([folder '\' filelist(file).name]);
    data = [data wave_data];
    fs(file) =  wave_fs;
end

end