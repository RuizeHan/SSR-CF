% function results = run_CCOTSSR()
% function results = run_CCOTSSR040(seq, res_path, bSaveImage, parameters)
function CCOTSSR
setup_paths();
bSaveImage = 0;
% video_path = 'sequences/Hiding';
% [seq, ~] = load_video_info(video_path);

close all;
seq.format = 'vot';
addpath(genpath('D:\tracking\VOT\vot-tir2016\workspace\trackers\CCOTSSR'));
addpath(genpath('./Processing'));
addpath(genpath('./networks'));
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));
% VOT: Get initialization data 数据的初始化
[handle, image_path, region] = vot('rectangle');
im = imread(image_path);

%% 参数设置&&数据准备
paramsCSR = [];
dataCSR = [];
params = SetParams(region);
[params, data] = PrepareData(params,im);

frame = 0;

while true
    
    frame = frame + 1;
    data.frame = frame;
    dataCSR.frame = frame;
    
    [handle, image_path] = handle.frame(handle);
    
    if isempty(image_path)
        break;
    end
    
    data.im = imread(image_path);
    if size(data.im,3) > 1 && data.colorImage == false
        data.im = data.im(:,:,1);
    end
    
    if params.useCSR == false
        [params, data] = Detection(params, data, 'FSR');
        if params.useCSR == false
            [params, data] = FilterUpdate(params,data);
        end
    end
    
    if params.enableCSR
        if params.useCSR == true
            if params.init == 0
%               seqCSR = seq;
%               seqCSR.init_rect = [data.pos(2), data.pos(1), data.target_sz(2), data.target_sz(1)];
                CSRregion(1) = double(data.pos(2) - data.target_sz(2)/2);
                CSRregion(2) = double(data.pos(1) - data.target_sz(1)/2);
                CSRregion(3) = double(data.target_sz(2));
                CSRregion(4) = double(data.target_sz(1));

                [paramsCSR] = SetParams(CSRregion);
                paramsCSR.useCSR = true;
                [paramsCSR, dataCSR] = PrepareData(paramsCSR,data.im);
                params.init = 1;
                paramsCSR.seq.frame = 1;
                dataCSR.im  = data.im;
                dataCSR.frame = 1;
            else
                [params, data] = Detection(params, data, 'FSR');
                [paramsCSR, dataCSR] = Detection(paramsCSR, dataCSR, 'CSR');
            end
            [paramsCSR, dataCSR] = FilterUpdate(paramsCSR, dataCSR);
        end
    end
    
%   Visualization(bSaveImage, params.useCSR, frame, data.im, data.pos, data.target_sz);
    
    Visualization(bSaveImage, params.useCSR, data.frame, data.im, data.pos, data.target_sz);
    
    nregion(1) = double(data.pos(2) - data.target_sz(2)/2);
    nregion(2) = double(data.pos(1) - data.target_sz(1)/2);
    nregion(3) = double(data.target_sz(2));
    nregion(4) = double(data.target_sz(1));
    
    handle = handle.report(handle, nregion);
        
end

% elapsed_time = toc;
% results.type = 'rect';
% results.res = data.OTB_rect_positions;
% results.fps = data.num_frames/(elapsed_time);
% fclose('all');

handle.quit(handle);

end
