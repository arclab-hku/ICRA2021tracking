% GET_FEATURES: Extracting hierachical convolutional features

function [feat, distictive_score, idx] = ran_get_features(im, cos_window, layers, target_mask)

global net
global enableGPU
% layers_number = layers(1);

if isempty(net)
      ran_initial_net(max(layers));
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
feat = cell(length(layers), 1);
idx = cell(length(layers), 1);
distictive_score = cell(length(layers), 1);

for ii = 1:length(layers)
    % Resize to sz_window
    if enableGPU
        x = gather(res(layers(ii)).x); 
    else
        x = res(layers(ii)).x;
    end
    
    target_mask = imResample(target_mask, size(x(:,:,1)));
    distictive_score{ii} = getDistictiveScore(x, target_mask);
%     idx = find(distictive_score > 0.1);
%     x = x(:,:,idx);
%     [sorted_score, index] = sort(distictive_score{ii}, 'descend');
%     idx{ii} = index(1:100);
    idx{ii} = find(distictive_score{ii} > 0);
%     x = x(:,:,idx{ii});

%     figure(2);
%     clf;
%     for jj = 1:length(idx{ii})
%         imagesc(x(:,:,idx{ii}(jj)));
%         xlabel(['layer idx = ', num2str(layers(ii))]);
%         colormap('jet');
%         pause;
%     end
    
    x = imResample(x, sz_window(1:2));
    
    % windowing technique
    if ~isempty(cos_window),
        x = bsxfun(@times, x, cos_window);
    end
    
    feat{ii}=x;
    
end

end
