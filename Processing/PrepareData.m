function [params, data] = PrepareData(params)

data.obj = [];
data.objf = [];
data.conf = [];
data.seq = [];
data.setup =[];
data.reg = [];

if exist(params.svm_path)
    load(params.svm_path);
    data.reg.model = svmmodel;
end

data.obj.pos = floor(params.init_pos);
data.obj.pos_prev = data.obj.pos;
data.obj.target_sz = floor(params.wsize);
data.seq.num_frames = numel(params.s_frames);
init_target_sz = data.obj.target_sz;
data.setup.featureRatio = params.t_global.cell_size;

if ~isfield(params, 'interpolation_method')
    params.interpolation_method = 'none';
end
if ~isfield(params, 'interpolation_centering')
    params.interpolation_centering = false;
end
if ~isfield(params, 'interpolation_windowing')
    params.interpolation_windowing = false;
end
if ~isfield(params, 'clamp_position')
    params.clamp_position = false;
end

% Calculate feature dimension
im = imread(params.s_frames{1});
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        data.seq.colorImage = false;
    else
        data.seq.colorImage = true;
    end
else
   data.seq.colorImage = false;
end

% use the maximum feature ratio right now todetermin the search size
search_area = prod(init_target_sz * params.search_area_scale);

if search_area > params.max_image_sample_size
    data.obj.currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    data.obj.currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    data.obj.currentScaleFactor = 1.0;
end

% target size at the initial scale
data.obj.base_target_sz = data.obj.target_sz / data.obj.currentScaleFactor;

%window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        data.obj.sz = floor( data.obj.base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        data.obj.sz = repmat(sqrt(prod(data.obj.base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        data.obj.sz = data.obj.base_target_sz + sqrt(prod(data.obj.base_target_sz * params.search_area_scale) + (data.obj.base_target_sz(1) - data.obj.base_target_sz(2))/4) - sum(data.obj.base_target_sz)/2; % const padding
    case 'custom'
        data.obj.sz = [data.obj.base_target_sz(1)*2 data.obj.base_target_sz(2)*4]; % for testing
end

[params.t_features, params.t_global, feature_info] = ...
    init_features(params.t_features, params.t_global, data.seq.colorImage, data.obj.sz, 'odd_cells');

% Set feature info
data.setup.img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
data.setup.feature_dim = feature_info.dim;
data.setup.num_feature_blocks = length(data.setup.feature_dim);
data.setup.feature_reg = permute(num2cell(feature_info.penalty), [2 3 1]);

% Size of the extracted feature maps
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,data.setup.num_feature_blocks), 2), [2 3 1]);

% Number of Fourier coefficients to save for each filter layer. This will
% be an odd number.
data.setup.filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(data.setup.filter_sz, ones(1,data.setup.num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size.
data.setup.output_sz = max(data.setup.filter_sz, [], 1);

% How much each feature block has to be padded to the obtain output_sz
data.setup.pad_sz = cellfun(@(filter_sz) (data.setup.output_sz - filter_sz) / 2, filter_sz_cell, 'uniformoutput', false);

% Compute the Fourier series indices and their transposes
ky = circshift(-floor((data.setup.output_sz(1) - 1)/2) : ceil((data.setup.output_sz(1) - 1)/2), [1, -floor((data.setup.output_sz(1) - 1)/2)])';
kx = circshift(-floor((data.setup.output_sz(2) - 1)/2) : ceil((data.setup.output_sz(2) - 1)/2), [1, -floor((data.setup.output_sz(2) - 1)/2)]);
data.setup.ky_tp = ky';
data.setup.kx_tp = kx';

% construct the Gaussian label function using Poisson summation formula
% sig_y = sqrt(prod(floor(base_target_sz))) * params.output_sigma_factor * (output_sz ./ img_sample_sz);
sig_y = sqrt(prod(floor(data.obj.base_target_sz))) * params.output_sigma_factor * (data.setup.output_sz ./ data.setup.img_support_sz);
yf_y = single(sqrt(2*pi) * sig_y(1) / data.setup.output_sz(1) * exp(-2 * (pi * sig_y(1) * ky / data.setup.output_sz(1)).^2));
yf_x = single(sqrt(2*pi) * sig_y(2) / data.setup.output_sz(2) * exp(-2 * (pi * sig_y(2) * kx / data.setup.output_sz(2)).^2));
y_dft = yf_y * yf_x;

% Compute the labelfunction at the filter sizes
data.setup.yf = cellfun(@(sz) fftshift(resizeDFT2(y_dft, sz, false)), filter_sz_cell, 'uniformoutput', false);
data.setup.yf = compact_fourier_coeff(data.setup.yf);

% construct cosine window
data.setup.cos_window = cellfun(@(sz) single(hann(sz(1)+2)*hann(sz(2)+2)'), feature_sz_cell, 'uniformoutput', false);
data.setup.cos_window = cellfun(@(cos_window) cos_window(2:end-1,2:end-1), data.setup.cos_window, 'uniformoutput', false);

% Compute Fourier series of interpolation function
[data.setup.interp1_fs, data.setup.interp2_fs] = cellfun(@(sz) get_interp_fourier(sz, params), filter_sz_cell, 'uniformoutput', false);

% Get the reg_window_edge parameter
reg_window_edge = {};
for k = 1:length(params.t_features)
    if isfield(params.t_features{k}.fparams, 'reg_window_edge')
        reg_window_edge = cat(3, reg_window_edge, permute(num2cell(params.t_features{k}.fparams.reg_window_edge(:)), [2 3 1]));
    else
        reg_window_edge = cat(3, reg_window_edge, cell(1, 1, length(params.t_features{k}.fparams.nDim)));
    end
end

% Construct spatial regularization filter
data.objf.filter = cellfun(@(reg_window_edge) get_reg_filter(data.setup.img_support_sz, data.obj.base_target_sz, params, reg_window_edge), reg_window_edge, 'uniformoutput', false);

if params.enableCSR
    [~, ~, d3] = size(data.objf.filter);
    for id = 1:d3
        rf = data.objf.filter{:,:,id};
        max_rf = max(max(rf));
        data.conf.filter{:,:,id} = max_rf - rf;
    end
end

% Compute the energy of the filter (used for preconditioner)
data.objf.energy = cellfun(@(reg_filter) real(reg_filter(:)' * reg_filter(:)), data.objf.filter, 'uniformoutput', false);
data.conf.energy = cellfun(@(con_filter) real(con_filter(:)' * con_filter(:)), data.conf.filter, 'uniformoutput', false);

if params.number_of_scales > 0
    scale_exp = (-floor((params.number_of_scales-1)/2):ceil((params.number_of_scales-1)/2));
    data.setup.scaleFactors = params.scale_step .^ scale_exp;
    %force reasonable scale changes
    data.setup.min_scale_factor = params.scale_step ^ ceil(log(max(5 ./ data.setup.img_support_sz)) / log(params.scale_step));
    data.setup.max_scale_factor = params.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ data.obj.base_target_sz)) / log(params.scale_step));
end

% Initialize and allocate
data.setup.prior_weights = [];
data.setup.sample_weights = [];
data.setup.latest_ind = [];
data.setup.samplesf = cell(1, 1, data.setup.num_feature_blocks);
for k = 1:data.setup.num_feature_blocks
    data.setup.samplesf{k} = complex(zeros(params.nSamples,data.setup.feature_dim(k),data.setup.filter_sz(k,1),(data.setup.filter_sz(k,2)+1)/2,'single'));
end

end