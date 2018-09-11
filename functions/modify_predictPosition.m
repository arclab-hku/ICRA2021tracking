function pos = ran_predictPosition(feat, pos, indLayers, nweights, cell_size, l1_patch_num, ...
    model_xf, model_alphaf, distictive_score)

% ================================================================================
% Compute correlation filter responses at each layer
% ================================================================================
res_layer = zeros([l1_patch_num, length(indLayers)]);

for ii = 1 : length(indLayers)
    if ~isempty(feat{ii})
        zf = fft2(feat{ii});
        featuremap_weight = distictive_score{ii};
        featuremap_weight(featuremap_weight < 0) = 0;
%         featuremap_weight(featuremap_weight > 0) = 1;
        featuremap_weight  = reshape(featuremap_weight,1,1,[]);
        weighted_kzf = bsxfun(@times, zf .* conj(model_xf{ii}), featuremap_weight);
        kzf=sum(weighted_kzf, 3) / numel(zf);
        res_layer(:,:,ii) = real(fftshift(ifft2(model_alphaf{ii} .* kzf)));  %equation for fast detection
    end
end

% Combine responses from multiple layers (see Eqn. 5)
response = sum(bsxfun(@times, res_layer, nweights), 3);

% ================================================================================
% Find target location
% ================================================================================
% Target location is at the maximum response. we must take into
% account the fact that, if the target doesn't move, the peak
% will appear at the top-left corner, not at the center (this isextractFeature
% discussed in the KCF paper). The responses wrap around cyclically.
[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
vert_delta  = vert_delta  - floor(size(zf,1)/2);
horiz_delta = horiz_delta - floor(size(zf,2)/2);

% Map the position to the image space
pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];

end