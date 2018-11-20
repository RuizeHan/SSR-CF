function training_rl()

clear
close all
base_path = 'C:\Users\ASUS\Desktop\tracker_benchmark_v1.0\trackers\CCOTSSR';
addpath(base_path);

pathAnno = 'C:\Users\ASUS\Desktop\tracker_benchmark_v1.0\anno\';
save_path = fullfile(base_path,'/training/train_results');
trainstate=1;
seqs = configSeqs_rl;

numSeq=length(seqs);
syn_s_len = 10;
for idxSeq =1:numSeq
    %     while trainstate ==0
    s = seqs{idxSeq};
    rect_anno = dlmread([pathAnno s.name '.txt']);
    s.name = [s.name,'_1'];
%     if exist([s.name,'svmmodel.mat'])
%         load([s.name,'svmmodel.mat'])
%         load([s.name,'train_set.mat'])
%         load([s.name,'syn_s.mat'])
%     else
        svmmodel = [];
        train_set = [];
        syn_s=[];
%     end
    
    nz	= strcat('%0',num2str(s.nz),'d'); %number of zeros in the name of image
    s.len = s.endFrame-s.startFrame+1;
    for i=1:s.len
        image_no = s.startFrame + (i-1);
        id = sprintf(nz,image_no);
        s.s_frames{i} = strcat(s.path,'img/',id,'.',s.ext);
    end
    
    img = imread(s.s_frames{1});
    
    s.anno = rect_anno;
    s.init_rect = rect_anno(s.startFrame,:);
    
    if isempty(syn_s)
        syn_s = gen_syn_seq(s,syn_s_len);
        
        if ~isfield(s,'slt_frames')
            syn_s.slt_frames = zeros(syn_s.len,1);
        end
        %
        if ~isfield(s,'max_tmp')
            syn_s.max_tmp = -Inf;
        end
    end
    
    [svmmodel,syn_s,train_set,trainstate] = train_CCOTSSR(syn_s,svmmodel,train_set);
    
    close all
    
    %         if trainstate
    save(fullfile(save_path,[s.name,'svmmodel.mat']),'svmmodel');
    save(fullfile(save_path,[s.name,'train_set.mat']),'train_set');
    save(fullfile(save_path,[s.name,'syn_s.mat']),'syn_s');
    %            break;
    %         end
    
    %     end
end
end

function [max_response,contrast] = cal_resfeat(response,objsz)
% calculte the maximum of response
max_response = max(response(:));
min_response = min(response(:));
contrast = (max_response-min_response).^2./mean((response(:)-min_response).^2);

end

function syn_s = gen_syn_seq(s,seq_len)
syn_s = s;
syn_s.len = seq_len;
syn_s.endFrame = syn_s.startFrame+seq_len-1;
syn_s.anno = zeros(seq_len,4);
syn_s.anno(1,:) = s.anno(1,:);

%     % generate random seqs
%     for si = 2:seq_len
%         % random occlusion
%         syn_s.occ_ports(si) = rand*0.3;
%         % random translation
%         syn_s.trans_pos(si,:) = [rand*0.5,rand*0.5];
%         % random occlusion position
%         syn_s.occ_pos(si,:) = [rand.*(1-sqrt(syn_s.occ_ports(si))),rand.*(1-sqrt(syn_s.occ_ports(si)))];
%     end

% generate random seqs
syn_s.occ_ports = zeros(seq_len,2);
syn_s.trans_pos(1,:)=[0,0];
syn_s.trans_pos(2,:)=[0,0];
syn_s.trans_pos(3,:)=[0,0];


for si = 3:seq_len
    %
    if mod(si,3)==1
        %
        syn_s.occ_ports(si,:) = [0,0];
        
        % random translation
        syn_s.trans_pos(si,:) = [0,0];%syn_s.trans_pos(si-1,:)+[(-0.5+rand*0.5)*0.05,(-0.5+rand*0.5)*0.05];
    elseif mod(si,3)==2
        syn_s.occ_ports(si,:) = [0.3+rand*0.5,0.3+rand*0.5];
        syn_s.trans_pos(si,:) = syn_s.trans_pos(si-1,:);
    else
        syn_s.occ_ports(si,:) = [0,0];
        syn_s.trans_pos(si,:) = syn_s.trans_pos(si-1,:);
    end
    % random occlusion position
    syn_s.occ_pos(si,:) = [0.25+rand.*(1-syn_s.occ_ports(si,1)),0.25+rand.*(1-syn_s.occ_ports(si,2))];
    syn_s.occ_bg_pos(si) = rand;
    % if add motion blur
    isblur=randi(2)-1;
    if isblur
        syn_s.blur_len(si) = randi([5,11]);
        syn_s.blur_deg(si) = randi([0,5]);
    else
        syn_s.blur_len(si) = 0;
        syn_s.blur_deg(si) = 0;
    end
end

switch s.name(1:end-2)
%     case 'fish'
%         syn_s.thr_small=0.05;
%         syn_s.thr_large=0.15;
%     case 'freeman3'
%         syn_s.thr_small=0.1;
%         syn_s.thr_large=0.25;    
%     case 'mountainbike'
%         syn_s.thr_small=0.04;
%         syn_s.thr_large=0.15; 
%     % fine-tune   
%     case 'lemming'
%         syn_s.thr_small=0.15;
%         syn_s.thr_large=0.15; 
%     case 'girl'
%         syn_s.thr_small=0.1;
%         syn_s.thr_large=0.25;    
%     case 'soccer'
%         syn_s.thr_small=0.02;
%         syn_s.thr_large=0.15; 
%         
%      case 'singer2'
%         syn_s.thr_small=0.04;
%         syn_s.thr_large=0.25;
%         
%      case 'freeman1'
%         syn_s.thr_small=0.04;
%         syn_s.thr_large=0.25;
%         
%      case 'freeman4'
%         syn_s.thr_small=0.04;
%         syn_s.thr_large=0.75;
%
%      case 'ironman'
%         syn_s.thr_small=0.04;
%         syn_s.thr_large=0.12;
        
    otherwise
        syn_s.thr_small=0.04;
        syn_s.thr_large=0.15;         
end


end
