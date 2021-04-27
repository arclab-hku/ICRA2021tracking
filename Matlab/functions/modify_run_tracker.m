% RUN_TRACKER: process a specified video using CF2
%
% Input:
%     - video:              the name of the selected video
%     - show_visualization: set to True for visualizing tracking results
% Output:
%
%
function [positions, fps, rects, highest_layer] = modify_run_tracker(video, base_path, show_visualization)

% Extra area surrounding the target
padding = struct('generic', 2, 'large', 1.5, 'height', 0.6);

lambda = 1e-4;              % Regularization parameter (see Eqn 3 in our paper)
output_sigma_factor = 0.1;  % Spatial bandwidth (proportional to the target size)

interp_factor = 0.01;       % Model learning rate (see Eqn 6a, 6b)
cell_size = 4;              % Spatial cell size

global enableGPU;
enableGPU = false;
        
[img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video);

% Call tracker function with all the relevant parameters
[positions, time, rects, highest_layer] = modify_tracker_ensemble(video_path, img_files, pos, target_sz, ...
    padding, lambda, output_sigma_factor, interp_factor, ...
    cell_size, show_visualization);

fps = numel(img_files) / time;

end
