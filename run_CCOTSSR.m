function results=run_CCOTSSR(seq, res_path, bSaveImage, parameters)

setup_paths();
close all

seq.format = 'otb';
params = SetParams(seq);
[params, data] = PrepareData(params);
time = 0;

for frame = 1:data.seq.num_frames
    data.seq.frame = frame;
%   frame
    data.seq.im = imread(params.s_frames{data.seq.frame});
    if size(data.seq.im,3) > 1 && data.seq.colorImage == false
        data.seq.im = data.seq.im(:,:,1);
    end
%
    tic();
    [params, data] = Detection(params, data);
    [params, data] = FilterUpdate(params, data);
    time = time + toc();
    data.seq.im_prev = data.seq.im;
    Visualization(bSaveImage, params.selector, data.seq.frame, data.seq.im, data.obj.pos, data.obj.target_sz);
%    
end


fps = numel(params.s_frames) / time;

% disp(['fps: ' num2str(fps)])

results.type = 'rect';
results.res = data.obj.rects;%each row is a rectangle
results.fps = fps;

end
