function [params, data] = Detection(params, data)
    % Do not estimate translation and scaling on the first frame, since we 
    % just want to initialize the tracker there
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
            % response_Jason = ifft2(scores_fs, 'symmetric');
            
           %% perform selector
            slt_response = scores_fs(:,:,scale_ind);
            if params.enableCSR && ~isempty(data.reg.model)
                params.selector  = cal_selector(data.reg.model,slt_response);
            end
            switch params.selector
                case -1 % context filter
                    params.count_c = params.count_c+1;
                    if params.count_c>params.count_thr
                        params.selector = 0;
                    end
                    params.update_c = 1;
                    [params, data] = FilterUpdate(params, data);
                    params.update_c = 0;
                case {1,0} % object filter
                    params.count_c = 0;
            end
            if params.selector ==-1
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
            data.obj.pos_prev = old_pos;
            data.obj.pos = data.obj.pos + translation_vec;
            data.obj.currentScaleFactor_prev = oldcurrentScaleFactor;
            
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

function [max_scale_response,contrast] = cal_resfeat(response)
% calculte the maximum of response
max_scale_response = max(response(:));
min_response = min(response(:));
contrast = (max_scale_response-min_response).^2./mean((response(:)-min_response).^2);
end