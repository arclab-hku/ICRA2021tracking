clc;
clear all;
close all;

addpath('functions','model','external/matconvnet/matlab');
vl_setupnn();
vl_compilenn();

% Extra area surrounding the target
padding = struct('generic', 2, 'large', 1.5, 'height', 0.6);
lambda = 1e-4;              % Regularization parameter (see Eqn 3 in our paper)
output_sigma_factor = 0.1;  % Spatial bandwidth (proportional to the target size)
interp_factor = 0.01;       % Model learning rate (see Eqn 6a, 6b)
cell_size = 4;              % Spatial cell size
global enableGPU;
enableGPU = false;
show_visualization = 1;

data_path = '.\uav_data\Gonzen_day2_2\';
img_files = dir(fullfile(data_path,'*.jpg'));
img_list = sort({img_files.name});
frame_id = 1;
img = imread(fullfile(data_path,img_list{1}));
im_sz = size(img);

try 
    ground_truth = importdata([data_path,'initial_rect.txt']);
catch
    figure(1);
    imshow(img);
    title('selecting target');
    ground_truth = getrect;
    ground_truth = round(ground_truth);
end
target_sz = [ground_truth(1,4), ground_truth(1,3)];
pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);
window_sz = get_search_window(target_sz, im_sz, padding);
ground_truth

[positions, time, rects, highest_layer] = modify_tracker_ensemble(data_path, img_list, pos, target_sz, ...
    padding, lambda, output_sigma_factor, interp_factor, ...
    cell_size, show_visualization);

