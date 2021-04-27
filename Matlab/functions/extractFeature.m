function feat = extractFeature(im, pos, window_sz, cos_window, indLayers)
% Get the search window from previous detection
patch = get_subwindow(im, pos, window_sz);
% Extracting hierarchical convolutional features
feat = get_features(patch, cos_window, indLayers);
end