function fm_std = ran_stdUpdate(target_map)

% patch = imResample(searching_patch, size(target_map));
% I = rgb2gray(patch);
% Iedge = edge(I,'Canny');
% Iedge = double(Iedge);
% map = bsxfun(@times, double(I), target_map);
[row, col, weights] = find(target_map);
if length(weights) > 10
    std_row = std(row,weights);
    std_col = std(col,weights);
    fm_std = [std_row, std_col];
else
    fm_std = [0, 0];
end

return