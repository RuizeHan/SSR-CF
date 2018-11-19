function [params,data,train_set,trainstate,max_tmp,slt_frames] = Detection_trainRL(params, data, max_tmp,gt_rect,train_set,slt_frames,thr_1,thr_2)

trainstate =0;
if data.seq.frame > 1
    old_pos = inf(size(data.obj.pos));
    iter = 1;
    
    %translation search
    while iter <= params.refinement_iterations && any(old_pos ~= data.obj.pos)
        % Extract features at multiple resolutions
        xt = extract_features(data.seq.im, data.obj.pos, data.obj.currentScaleFactor*data.setup.scaleFactors, params.t_features, params.t_global);
        
        % Do windowing of features
        xt = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, data.setup.cos_window, 'uniformoutput', false);
        
        % Compute the fourier series
        xtf = cellfun(@cfft2, xt, 'uniformoutput', false);
        
        % Interpolate features to the continuous domain
        xtf = interpolate_dft(xtf, data.setup.interp1_fs, data.setup.interp2_fs);
        
        [~,~, scale] = size(data.setup.pad_sz);
        for sc = 1:scale
            data.setup.pad_sz{:,:,sc} = double( data.setup.pad_sz{:,:,sc});
        end
        
        % Compute convolution for each feature block in the Fourier domain
        scores_fs_feat = cellfun(@(hf, xf, pad_sz) padarray(sum(bsxfun(@times, hf, xf), 3), pad_sz), data.objf.hf_full, xtf, data.setup.pad_sz, 'uniformoutput', false);
        
        % Also sum over all feature blocks.
        % Gives the fourier coefficients of the convolution response.
        scores_fs = ifftshift(ifftshift(permute(sum(cell2mat(scores_fs_feat), 3), [1 2 4 3]), 1), 2);
        
        % Optimize the continuous score function with Newton's method.
        [max_scale_response, trans_row, trans_col, scale_ind] = optimize_scores(scores_fs, params.newton_iterations, data.setup.ky_tp, data.setup.kx_tp);
            
        %% perform selector
        slt_response = scores_fs(:,:,scale_ind);
        if params.enableCSR && ~isempty(data.reg.model)
            params.selector  = cal_selector(data.reg.model,slt_response);
        end
        
        %% perform detection
        switch params.selector
            case -1 % context filter
                params.update_c = 1;
                [params, data] = FilterUpdate(params, data);
                filters = data.conf;
                params.update_c = 0;
            case 1 % object filter
                filters = data.objf;
        end
        
        if params.selector==-1
            % Compute convolution for each feature block in the Fourier domain
            scores_fs_feat = cellfun(@(hf, xf, pad_sz) padarray(sum(bsxfun(@times, hf, xf), 3), pad_sz), data.conf.hf_full, xtf, data.setup.pad_sz, 'uniformoutput', false);
            
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = ifftshift(ifftshift(permute(sum(cell2mat(scores_fs_feat), 3), [1 2 4 3]), 1), 2);
            
            % Optimize the continuous score function with Newton's method.
            [max_scale_response, trans_row, trans_col, scale_ind] = optimize_scores(scores_fs, params.newton_iterations, data.setup.ky_tp, data.setup.kx_tp);
        end
        % Compute the translation vector in pixel-coordinates and round
        % to the closest integer pixel.
        translation_vec = round([trans_row, trans_col] .* (data.setup.img_support_sz./data.setup.output_sz) * data.obj.currentScaleFactor * data.setup.scaleFactors(scale_ind));
        
        % set the scale
        oldcurrentScaleFactor = data.obj.currentScaleFactor;
        data.obj.currentScaleFactor = data.obj.currentScaleFactor * data.setup.scaleFactors(scale_ind);
        
        % adjust to make sure we are not to large or to small
        if data.obj.currentScaleFactor < data.setup.min_scale_factor
            data.obj.currentScaleFactor = data.setup.min_scale_factor;
        elseif data.obj.currentScaleFactor > data.setup.max_scale_factor
            data.obj.currentScaleFactor = data.setup.max_scale_factor;
        end
        
        % update position
        old_pos = data.obj.pos;
        data.obj.pos = data.obj.pos + translation_vec;
        data.obj.pos_prev = old_pos;
        data.obj.currentScaleFactor_prev = oldcurrentScaleFactor;
        
        % calcalute the ground truth action
        gt_pos = [gt_rect(2)+gt_rect(4)./2,gt_rect(1)+gt_rect(3)./2];
        cle = norm(gt_pos-data.obj.pos);
        
        % if the regression model get gt action
        if cle<1 && max_scale_response>max_tmp
            max_tmp = max_scale_response;
        end
        
        if max_scale_response< thr_1%&& params.selector~=-1 %% cle>5
            if ~slt_frames(data.seq.frame)
                num_sample = numel(train_set);
                train_set(num_sample+1).response = ifftshift(slt_response);
                train_set(num_sample+1).gt_sltor = -1;
                train_set(num_sample+1).objsz = round(data.obj.target_sz./data.setup.featureRatio);
                slt_frames(data.seq.frame)=1;
                for ti=1:numel(train_set)
                    gt_data(ti,:) = train_set(ti).gt_sltor;
                end
                classes = unique(gt_data);
                if numel(classes)>1
                    data.reg.model = train_svm(train_set);
                    %                     trainstate = 1;
                end
            end
        elseif max_scale_response> thr_2 && cle<=2 %0.5*max_tmp
            if ~slt_frames(data.seq.frame)
                num_sample = numel(train_set);
                train_set(num_sample+1).response = ifftshift(slt_response);
                train_set(num_sample+1).gt_sltor = 0;
                train_set(num_sample+1).objsz = round(data.obj.target_sz./data.setup.featureRatio);
                slt_frames(data.seq.frame)=1;
                for ti=1:numel(train_set)
                    gt_data(ti,:) = train_set(ti).gt_sltor;
                end
                classes = unique(gt_data);
                if numel(classes)>1
                    data.reg.model = train_svm(train_set);
                    %                     trainstate = 1;
                end
            end
        elseif max_scale_response>=thr_1 && max_scale_response<=thr_2 %&& cle<1%&& params.selector~=1
            if ~slt_frames(data.seq.frame)
                num_sample = numel(train_set);
                train_set(num_sample+1).response = ifftshift(slt_response);
                train_set(num_sample+1).gt_sltor = 1;
                train_set(num_sample+1).objsz = round(data.obj.target_sz./data.setup.featureRatio);
                slt_frames(data.seq.frame)=1;
                for ti=1:numel(train_set)
                    gt_data(ti,:) = train_set(ti).gt_sltor;
                end
                classes = unique(gt_data);
                if numel(classes)>1
                    data.reg.model = train_svm(train_set);
                    %                     trainstate = 1;
                end
            end
        end
        
        
        if params.clamp_position
            data.obj.pos = max([1 1], min([size(im,1) size(im,2)], data.obj.pos));
        end
        data.obj.target_sz = floor(data.obj.base_target_sz * data.obj.currentScaleFactor);
        %save position and calculate FPS
        data.obj.rects(data.seq.frame,:) = [data.obj.pos([2,1]) - floor(data.obj.target_sz([2,1])/2), data.obj.target_sz([2,1])];
        
        iter = iter + 1;
    end
end

end

function selector = cal_selector(svmmodel,response)

[max_scale_response,contrast] = cal_resfeat(response);
scores = zeros(3,1);
for si=1:numel(svmmodel)
    if ~isempty(svmmodel{si})
        [~,score]=predict(svmmodel{si},[max_scale_response,contrast]);
        scores(si) = score(2);
    end
end

[~,selector] = max(scores);
if scores(1)==scores(2)
    selector = 2;
end
selector = selector -2;
end

function svmmodels = train_svm(train_set)

data = [];
gt_data = [];
svmmodels = cell(3,1);
for ti=1:numel(train_set)
    response = train_set(ti).response;
    objsz = train_set(ti).objsz;
    [data(ti,1),data(ti,2)] = cal_resfeat(response);
    gt_data(ti,:) = train_set(ti).gt_sltor+2; % 1: context 2: no update 3: foreground
end

classes = unique(gt_data);
for ci = 1:numel(classes)
    gt_ci = (gt_data==classes(ci));
    svmmodels{classes(ci)} = fitcsvm(data,gt_ci,'KernelFunction','gaussian','Standardize',true,...
        'BoxConstraint',Inf,'ClassNames',[false true]);
end

if 0
    %% debug show svm results
    d2 = 1; d1 = 0.01;
    [x1Grid,x2Grid] = meshgrid(min(data(:,1)):d1:max(data(:,1)),...
        min(data(:,2)):d2:max(data(:,2)));
    xGrid = [x1Grid(:),x2Grid(:)];
    N = size(xGrid,1);
    Scores = zeros(N,numel(classes));
    
    for j = classes';
        [~,score] = predict(svmmodels{j},xGrid);
        Scores(:,j) = score(:,2); % Second column contains positive-class scores
    end
    
    [~,maxScore] = max(Scores,[],2);
    
    figure(2)
    h(classes') = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
        [0.1 0.5 0.5; 0.5 0.1 0.5; 0.5 0.5 0.1]);
    hold on
    h(classes'+3) = gscatter(data(:,1),data(:,2),gt_data);
    title('{\bf Iris Classification Regions}');
    xlabel('Petal Length (cm)');
    ylabel('Petal Width (cm)');
    legend(h,{'setosa region','versicolor region','virginica region',...
        'observed setosa','observed versicolor','observed virginica'},...
        'Location','Northwest');
    axis tight
    hold off
end

end

function [max_scale_response,contrast] = cal_resfeat(response)
% calculte the maximum of response
max_scale_response = max(response(:));
min_response = min(response(:));
contrast = (max_scale_response-min_response).^2./mean((response(:)-min_response).^2);
end