function [model_xf, model_alphaf] = ran_updateModel(feat, yf, interp_factor, lambda, frame, ...
    model_xf, model_alphaf, distictive_score)

numLayers = length(feat);

% ================================================================================
% Initialization
% ================================================================================
xf       = cell(1, numLayers);
alphaf   = cell(1, numLayers);

% ================================================================================
% Model update
% ================================================================================
for ii=1 : numLayers
    xf{ii} = fft2(feat{ii});
    featuremap_weight = distictive_score{ii};
    featuremap_weight(featuremap_weight < 0) = 0;
%     featuremap_weight(featuremap_weight > 0) = 1;
    featuremap_weight  = reshape(featuremap_weight,1,1,[]);
    weighted_kf = bsxfun(@times, xf{ii} .* conj(xf{ii}), featuremap_weight);
    kf = sum(weighted_kf, 3) / numel(xf{ii});
    alphaf{ii} = yf./ (kf+ lambda);   % Fast training
end

% Model initialization or update
if frame == 1,  % First frame, train with a single image
    for ii=1:numLayers
        model_alphaf{ii} = alphaf{ii};
        model_xf{ii} = xf{ii};
    end
else
    % Online model update using learning rate interp_factor
    for ii=1:numLayers
        model_alphaf{ii} = (1 - interp_factor) * model_alphaf{ii} + interp_factor * alphaf{ii};
        model_xf{ii}     = (1 - interp_factor) * model_xf{ii}     + interp_factor * xf{ii};
    end
end


end