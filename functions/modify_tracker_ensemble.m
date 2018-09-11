% tracker_ensemble: Correlation filter tracking with convolutional features
%
% Input:
%   - video_path:          path to the image sequence
%   - img_files:           list of image names
%   - pos:                 intialized center position of the target in (row, col)
%   - target_sz:           intialized target size in (Height, Width)
% 	- padding:             padding parameter for the search area
%   - lambda:              regularization term for ridge regression
%   - output_sigma_factor: spatial bandwidth for the Gaussian label
%   - interp_factor:       learning rate for model update
%   - cell_size:           spatial quantization level
%   - show_visualization:  set to True for showing intermediate results
% Output:
%   - positions:           predicted target position at each frame
%   - time:                time spent for tracking
%
%   It is provided for educational/researrch purpose only.


function [positions, time, rects, highest_layer] = modify_tracker_ensemble(video_path, img_files, pos, target_sz, ...
    padding, lambda, output_sigma_factor, interp_factor, cell_size, show_visualization)

% ================================================================================
% CNN setting
% ================================================================================
layers_candidate = [19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37];
topNlayer = 2;
indLayers = layers_candidate(end); % test
nweights  = 1; % test
numLayers = length(indLayers);
foreground_extend_range = 0.25;

% ================================================================================
% Tracker setting
% ================================================================================
% Get image size and search window size
im_sz     = size(imread([video_path img_files{1}]));
window_sz = get_search_window(target_sz, im_sz, padding);

% Compute the sigma for the Gaussian function label
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
l1_patch_num = floor(window_sz/ cell_size);

% Pre-compute the Fourier Transform of the Gaussian function label
yf = fft2(gaussian_shaped_labels(output_sigma, l1_patch_num));

% Pre-compute and cache the cosine window (for avoiding boundary discontinuity)
cos_window = hann(size(yf,1)) * hann(size(yf,2))';

% Initialize variables for calculating FPS and distance precision
time      = 0;
positions = zeros(numel(img_files), 2);
nweights  = reshape(nweights,1,1,[]);

% Note: variables ending with 'f' are in the Fourier domain.
model_xf     = cell(1, numLayers);
model_alphaf = cell(1, numLayers);

rects = [];

% ================================================================================
% Scale estimation setting
% ================================================================================
featuremap_original = [];
memory_length = 5;
featuremap_array = zeros([l1_patch_num,memory_length]);
change_rate_array = ones([memory_length,2]);
featuremap_array_counter = 1;
original_std = [];
estimated_sz = target_sz;
estimated_window = window_sz;
change_rate_m = [1,1];
scale_alpha = 0.9;

figure(1);

% ================================================================================
% Start tracking
% ================================================================================
for frame = 1:numel(img_files)

    im = imread([video_path img_files{frame}]); % Load the image at the current frame
    if ismatrix(im)
        im = cat(3, im, im, im);
    end
    
    tic();
    % ================================================================================
    % Predicting the object position from the recommended feature maps
    % ================================================================================
    if frame > 1
        feat = extractFeature(im, pos, estimated_window, cos_window, indLayers);  
        for ii = 1:length(indLayers)
            if ~isempty(idx{ii})
                recommended_model_xf{ii} = model_xf{ii}(:,:,idx{ii});
                recommended_feat{ii} = feat{ii}(:,:,idx{ii});
                recommended_score{ii} = distictive_score{ii}(idx{ii});
            else
                recommended_model_xf{ii} = model_xf{ii};
                recommended_feat{ii} = feat{ii};
                recommended_score{ii} = distictive_score{ii};
            end
        end
        pos  = modify_predictPosition(recommended_feat, pos, indLayers, nweights, cell_size, l1_patch_num, ...
            recommended_model_xf, model_alphaf, recommended_score);
    else
        [patch, target_mask] = modify_get_subwindow(im, pos, estimated_window, target_sz);
        [layer_score, layer_index] = modify_get_layer(patch, cos_window, layers_candidate, target_mask);
        indLayers = layers_candidate(layer_index(1:topNlayer));
        nweights = layer_score(1:topNlayer);
        if nweights(1) <= 0
            nweights(1) = 1;
        end
        nweights = nweights./(nweights(1));
        numLayers = length(indLayers);
        model_xf     = cell(1, numLayers);
        model_alphaf = cell(1, numLayers);
        nweights  = reshape(nweights,1,1,[]);
    end
    
    highest_layer = max(indLayers);
    
    % ================================================================================
    % Learning correlation filters over hierarchical convolutional features
    % ================================================================================

    [feat, distictive_score, idx, featuremap, searching_patch] = modify_extractFeature(im, pos, estimated_sz, estimated_window, cos_window, indLayers, foreground_extend_range);
    % Model update
    [model_xf, model_alphaf] = modify_updateModel(feat, yf, interp_factor, lambda, frame, ...
        model_xf, model_alphaf,distictive_score);
    
    % ================================================================================
    % Learning scale over convolutional feature maps
    % ================================================================================
    featuremap_array(:,:,featuremap_array_counter) = featuremap;
    featuremap_array_counter = featuremap_array_counter + 1;
    if featuremap_array_counter > 5;
        featuremap_array_counter = 1;
    end
    target_map = sum(featuremap_array,3)./memory_length; 
    fm_std = modify_stdUpdate(target_map);
    if ~isempty(featuremap_original)
        sumRatio = sum(sum(target_map))/sum(sum(featuremap_original));
        change_rate = change_rate_m.*[scale_alpha*sumRatio, scale_alpha*sumRatio] + (1 - scale_alpha).*fm_std./original_std;
        change_rate_array = [change_rate_array;change_rate];
        change_rate_array(1,:) = [];
        change_rate_m = mean(medfilt1(change_rate_array));
        % ================================================================================
        % maximum and minimum scale change rate 
        % ================================================================================
        if change_rate_m(1) < 0.2
            change_rate_m(1) = 0.2;
        end
        if change_rate_m(1) > 5
            change_rate_m(1) = 5;
        end
        if change_rate_m(2) < 0.2
            change_rate_m(2) = 0.2;
        end
        if change_rate_m(2) > 5
            change_rate_m(2) = 5;
        end
        estimated_sz = round(change_rate_m.*target_sz);
        estimated_window = round(change_rate_m.*window_sz);
        
    else
        if frame > memory_length
            featuremap_original = target_map;
            original_std = fm_std;
        end
    end
    
    % ================================================================================
    % Save predicted position and timing
    % ================================================================================
    positions(frame,:) = pos;
    time = time + toc();
    
    % ================================================================================
    % Visualization
    % ================================================================================
    box = [pos([2,1]) - estimated_sz([2,1])/2, estimated_sz([2,1])];
    rects = [rects;box];
    if show_visualization,
        clf;
        imshow(im);
        hold on;
        rectangle('Position', box, 'EdgeColor', 'g', 'LineWidth', 1);
        xlabel(['frame = ',num2str(frame)]);
        hold off;
        drawnow;
    end
end

end

function feat = extractFeature(im, pos, window_sz, cos_window, indLayers)
% Get the search window from previous detection
patch = get_subwindow(im, pos, window_sz);
% Extracting hierarchical convolutional features
feat = get_features(patch, cos_window, indLayers);
end



