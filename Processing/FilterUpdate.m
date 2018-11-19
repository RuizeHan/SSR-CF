function [params, data] = FilterUpdate(params, data)

% Update the weights
[data.setup.prior_weights, replace_ind] = update_prior_weights(data.setup.prior_weights, data.setup.sample_weights, data.setup.latest_ind, data.seq.frame, params);
data.setup.latest_ind = replace_ind;
data.setup.sample_weights = data.setup.prior_weights;
%
if data.seq.frame ==1
    selector=1;
else
    selector= params.selector;
end
%
if ~params.update_c && selector==-1
    return 
end
%
switch selector
    case 1
        % Extract image region for training sample
        xl = extract_features(data.seq.im, data.obj.pos, data.obj.currentScaleFactor, params.t_features, params.t_global);
    case -1
        % Extract image region for training sample
        xl = extract_features(data.seq.im_prev, data.obj.pos_prev, data.obj.currentScaleFactor_prev, params.t_features,params.t_global);
    case 0
        return;
end

% Update the weights
% [data.prior_weights, replace_ind] = update_prior_weights(data.prior_weights, data.sample_weights, data.latest_ind, data.frame, params);
% data.latest_ind = replace_ind;
% data.sample_weights = data.prior_weights;

% Do windowing of features
xl = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, data.setup.cos_window, 'uniformoutput', false);

% Compute the fourier series
xlf = cellfun(@cfft2, xl, 'uniformoutput', false);

% Interpolate features to the continuous domain
xlf = interpolate_dft(xlf, data.setup.interp1_fs, data.setup.interp2_fs);

% New sample to be added
xlf = cellfun(@(xf) xf(:,1:(size(xf,2)+1)/2,:), xlf, 'uniformoutput', false);

% Insert the new training sample
for k = 1:data.setup.num_feature_blocks
    data.setup.samplesf{k}(replace_ind,:,:,:) = permute(xlf{k}, [4 3 1 2]);
end

% Construct the right hand side vector
rhs_samplef = cellfun(@(xf) permute(mtimesx(data.setup.sample_weights, 'T', xf, 'speed'), [3 4 2 1]), data.setup.samplesf, 'uniformoutput', false);
rhs_samplef = cellfun(@(xf, yf) bsxfun(@times, conj(xf), yf), rhs_samplef, data.setup.yf, 'uniformoutput', false);
new_sample_energy = cellfun(@(xlf) abs(xlf .* conj(xlf)), xlf, 'uniformoutput', false);

% learnning or updating context filter or target filter
switch selector
    case -1
        ftype = 'conf';       
    case 1
        ftype = 'objf';        
end
filter = getfield(data,ftype);

if data.seq.frame == 1
    % Initialize the filter
    filter.hf = cell(1,1,data.setup.num_feature_blocks);
    for k = 1:data.setup.num_feature_blocks
        filter.hf{k} = complex(zeros([data.setup.filter_sz(k,1) (data.setup.filter_sz(k,2)+1)/2 data.setup.feature_dim(k)], 'single'));
    end
    
    % Initialize Conjugate Gradient parameters
    filter.p = [];
    filter.rho = [];
    max_CG_iter = params.init_max_CG_iter;
    filter.sample_energy = new_sample_energy;
else
    max_CG_iter = params.max_CG_iter;
    
    if params.CG_forgetting_rate == inf || params.learning_rate >= 1
        % CG will be restarted
        filter.p = [];
        filter.rho = [];
    else
        filter.rho = filter.rho / (1-params.learning_rate)^params.CG_forgetting_rate;
    end
    
    % Update the approximate average sample energy using the learning
    % rate. This is only used to construct the preconditioner.
    filter.sample_energy = cellfun(@(se, nse) (1 - params.learning_rate) * se + params.learning_rate * nse, filter.sample_energy, new_sample_energy, 'uniformoutput', false);
end

% Construct preconditioner
diag_M = cellfun(@(m, reg_energy) (1-params.precond_reg_param) * bsxfun(@plus, params.precond_data_param * m, (1-params.precond_data_param) * mean(m,3))...
    + params.precond_reg_param*reg_energy, filter.sample_energy, filter.energy, 'uniformoutput',false);

% do conjugate gradient
[filter.hf, flag, relres, iter, res_norms, filter.p, filter.rho] = pcg_ccot(...
    @(x) lhs_operation(x, data.setup.samplesf, filter.filter, data.setup.sample_weights, data.setup.feature_reg),...
    rhs_samplef, params.CG_tol, max_CG_iter, ...
    @(x) diag_precond(x, diag_M), ...
    [], filter.hf, filter.p, filter.rho);

% Make the filter symmetric (avoid roundoff errors)
filter.hf = symmetrize_filter(filter.hf);

% Reconstruct the full Fourier series
filter.hf_full = full_fourier_coeff(filter.hf);

data = setfield(data,ftype,filter);
end