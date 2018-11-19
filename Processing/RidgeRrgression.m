function data = RidgeRrgression(data, params)



xt = extract_features(data.im, data.pos, data.currentScaleFactor*data.scaleFactors, params.t_features, data.global_fparams);

% Do windowing of features
xt = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, data.cos_window, 'uniformoutput', false);

% Compute the fourier series
xtf = cellfun(@cfft2, xt, 'uniformoutput', false);

% Interpolate features to the continuous domain
xtf = interpolate_dft(xtf, data.interp1_fs, data.interp2_fs);

% Compute convolution for each feature block in the Fourier domain
scores_fs_feat = cellfun(@(hf, xf, pad_sz) padarray(sum(bsxfun(@times, hf, xf), 3), pad_sz), data.hf_full, xtf, data.pad_sz, 'uniformoutput', false);

% Also sum over all feature blocks.
% Gives the fourier coefficients of the convolution response.
data.scores_fs = ifftshift(ifftshift(permute(sum(cell2mat(scores_fs_feat), 3), [1 2 4 3]), 1), 2);

[sz1, sz2, ~] = size(data.scores_fs);
output_sz = [sz1 sz2];

% Do the grid search step by finding the maximum in the sampled response
% for each scale.
sampled_scores = prod(output_sz) * ifft2(data.scores_fs, 'symmetric');

data.reg = jud_overlap([data.pos([2,1]), data.target_sz([2,1])], fftshift(sampled_scores(:,:,1)));

end