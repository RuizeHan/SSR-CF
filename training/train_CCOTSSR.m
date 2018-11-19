function [svmmodel,seq,train_set,trainstate] = train_CCOTSSR(seq,svmmodel,train_set)

setup_paths();
seq.format = 'otb';
params = SetParams(seq);
[params, data] = PrepareData(params);
trainstate = 1;
showImage = 1;
data.reg.model = svmmodel;
for frame = 1:seq.len
    
    data.seq.frame = frame;
    if frame==1
        first_img = imread(params.s_frames{data.seq.frame});
        data.seq.im = first_img;
        anno = seq.init_rect;
    else
        [data.seq.im,anno] = ...
            gen_syn_frame(first_img,seq.init_rect,seq.occ_ports(frame,:),...
            seq.occ_pos(frame,:),seq.occ_bg_pos(frame),...
            seq.trans_pos(frame,:),data.obj.sz,...
            seq.blur_len(frame),seq.blur_deg(frame));
    end
    
    if size(data.seq.im,3) > 1 && data.seq.colorImage == false
        data.seq.im = data.seq.im(:,:,1);
    end
    
    [params,data,train_set,trainstate,seq.max_tmp,seq.slt_frames] = ...
        Detection_trainRL(params,data,seq.max_tmp,anno,train_set,seq.slt_frames,seq.thr_small,seq.thr_large);
   
    [params, data] = FilterUpdate(params, data);
    
    data.seq.im_prev = data.seq.im;
    
    Visualization(showImage, params.selector, data.seq.frame, data.seq.im, data.obj.pos, data.obj.target_sz);
    
end

svmmodel = data.reg.model;
return

end

function [im,anno] = gen_syn_frame(first_img,rect,occ_port,occ_pos,occ_bg_pos,trans_port,sr_sz,blur_len,blur_deg)

     rows = floor(rect(2)+rect(4)/2-sr_sz(1)/2:rect(2)+rect(4)/2+sr_sz(1)/2);    
     cols = floor(rect(1)+rect(3)/2-sr_sz(2)/2:rect(1)+rect(3)/2+sr_sz(2)/2);  
     
     valid_rows = floor(numel(rows).* occ_port(1));
     valid_cols = floor(numel(cols).* occ_port(2));
     start_row = floor((numel(rows)-valid_rows).*occ_pos(1));
     start_col = floor((numel(cols)-valid_cols).*occ_pos(2));
     
     if start_row<1 
         start_row =1; 
     end
     if start_col<1 
         start_col =1; 
     end
     
     rows = rows(start_row:valid_rows+start_row-1);
     cols = cols(start_col:valid_cols+start_col-1);
     rows(rows<1) = [];
     cols(cols<1) = [];
     
     [xgrid,ygrid] = meshgrid(cols,rows);
     
     im_rows = [1:rect(2)-1,rect(4)+rect(2)-1:size(first_img,1)];
     im_cols = [1:rect(1)-1,rect(3)+rect(1)-1:size(first_img,2)];
     im_rows(im_rows<1) = [];
     im_cols(im_cols<1) = [];
     
     if numel(im_rows)<numel(rows)
        im_rows(end+1:numel(rows)) = im_rows(end);
        im_start_row = 1;
        im_end_row = numel(rows);
     else
        im_start_row = floor((numel(im_rows)-numel(rows)).*occ_bg_pos); 
        if im_start_row<1 
             im_start_row=1; 
        end
        im_end_row = numel(rows)+im_start_row-1; 
     end
     
     if numel(im_cols)<numel(cols)
         im_cols(end+1:numel(cols)) = im_cols(end);
         im_start_col = 1;
         im_end_col = numel(cols);
     else
         im_start_col = floor((numel(im_cols)-numel(cols)).*occ_bg_pos);
         if im_start_col<1 
             im_start_col=1; 
         end
         im_end_col = numel(cols)+im_start_col-1;
     end

     im_cols = im_cols(im_start_col:im_end_col);
     im_rows = im_rows(im_start_row:im_end_row);
     
     [im_xgrid,im_ygrid] = meshgrid(im_cols,im_rows);
     
     for ci = 1:size(first_img,3)
        c_img = first_img(:,:,ci);
        first_img(ygrid,xgrid,ci) = c_img(im_ygrid,im_xgrid);
     end
     
     trans_pos = floor(trans_port.*sr_sz);
     im = circshift(first_img,trans_pos);
     anno = [rect(1)+trans_pos(2),rect(2)+trans_pos(1),...
         rect(3),rect(4)];
     
     %% add motion blur
     if blur_len>0
         PSF = fspecial('motion', blur_len, blur_deg);
         im = imfilter(im, PSF, 'conv', 'circular');
     end
     
end