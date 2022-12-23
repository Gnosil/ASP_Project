function win_norm = win_norm(win,skip)
% !!!!! Extremely important for the process to converge
    win_norm = [win; zeros(skip*ceil(length(win)/skip)-length(win),1)];
    win_norm = reshape(win_norm,skip,[]);
    win_norm = win_norm./sqrt(sum(abs(win_norm).^2,2));
    win_norm = reshape(win_norm(1:length(win)),[],1);
end