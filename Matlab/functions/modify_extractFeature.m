function [feat, distictive_score, idx, featuremap, patch] = modify_extractFeature(im, pos, target_sz, window_sz, cos_window, indLayers, foreground_extend_range)
% Get the search window from previous detection
target_sz = round((1+ foreground_extend_range).*target_sz);
[patch, target_mask] = modify_get_subwindow(im, pos, window_sz, target_sz);
% Extracting hierarchical convolutional features
[feat, distictive_score, idx] = modify_get_features(patch, cos_window, indLayers, target_mask);

nfeat = length(feat);
img_max = sum(feat{1},3);

if nfeat > 1
    for ii = 2:nfeat;
        img = sum(feat{ii}(:,:,idx{ii}),3);
        img_max = bsxfun(@max, img_max, img);
%         figure(2);
%         clf;
%         imagesc(img);
%         colormap('jet');
%         pause;
    end
end

target_mask = imResample(target_mask,size(img_max));
img_max = bsxfun(@times, img_max, target_mask);
featuremap = img_max./(max(max(img_max)));
% featuremap(featuremap < 0.8) = 0;
% featuremap(featuremap > 0) = 1;

end