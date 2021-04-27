% GET_FEATURES: Extracting hierachical convolutional features

function [layer_score, layer_index] = modify_get_layer(im, cos_window, layers_candidate, target_mask)

global net
global enableGPU

if isempty(net)
      modify_initial_net(max(layers_candidate));
end

sz_window = size(cos_window);

% Preprocessing
img = single(im);        % note: [0, 255] range
img = imResample(img, net.normalization.imageSize(1:2));
img = img - net.normalization.averageImage;
if enableGPU, img = gpuArray(img); end

% Run the CNN
res = vl_simplenn(net,img);

% Initialize feature maps

for ii = 1:length(layers_candidate)
    % Resize to sz_window
    if enableGPU
        x = gather(res(layers_candidate(ii)).x); 
    else
        x = res(layers_candidate(ii)).x;
    end

    x = imResample(x, sz_window(1:2));
    
%     windowing technique
    if ~isempty(cos_window),
        x = bsxfun(@times, x, cos_window);
    end
    
    featuremap = sum(x,3);
    feat(:,:,ii) = featuremap;
    
end

target_mask = imResample(target_mask, sz_window(1:2));
distictive_score = getDistictiveScore(feat, target_mask);
[layer_score, layer_index] = sort(distictive_score, 'descend');

% figure(2);
% clf;
% for ii = 1:length(layers_candidate)
%     imagesc(feat(:,:,ii));
%     xlabel(['layer idx = ', num2str(layers_candidate(ii)), ' score = ', num2str(distictive_score(ii))]);
%     colormap('jet');
%     pause;
% end

end
