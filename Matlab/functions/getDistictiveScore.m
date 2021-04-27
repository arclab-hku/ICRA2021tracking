function distictive_score = getDistictiveScore(featuremap, target_mask)

maps_number = size(featuremap,3);
distictive_score = zeros([maps_number,1]);
im_size = size(target_mask);
im_center = im_size/2;
[r,c] = find(target_mask > 0.9,1,'first');
target_size = sqrt((r - im_center(1)).^2 + (c - im_center(2)).^2);

for ii = 1:maps_number
    
    A = featuremap(:,:,ii);
    
    normA = A - min(A(:));
    if(max(normA(:)) > 0)
        current_map = normA ./ max(normA(:));
        % background
        background_mask = ~target_mask; % binary map for background
        B = current_map.*background_mask; % get target featuremap
        Bmax = max(max(B));
        
        % foreground
        F = current_map.*target_mask; % get target featuremap
        Fmax = max(max(F));
        
        [row,col] = find(current_map == 1);
        
        dist = sqrt((row - im_center(1)).^2 + (col - im_center(2)).^2);

%         distictive_term = sum(-log(dist/target_size))/length(dist);
        distictive_term = sum(exp(1 - (dist/target_size).^2)-1)/length(dist);
        
        if isinf(distictive_term)
            disp('not a good idea...');
        end
        
        gain_term = (Fmax - Bmax)^2;
        distictive_score(ii) = distictive_term * gain_term;
    else
        distictive_score(ii) = 0;
    end
    
end

return