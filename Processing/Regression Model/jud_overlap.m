function reg = jud_overlap(init_pos, resp_shift)

start_pos = [round((200 - init_pos(3))/2), round((200 - init_pos(4))/2)];
init_pos(1:2) = [100,100];
[ctrows,ctrcols] = ndgrid(1:200,1:200);
resp = imresize(resp_shift, [200 200]);
Y = zeros(200*200,1);

rect(:,1) = ctrcols(:);
rect(:,2) = ctrows(:);
rect(:,3) = init_pos(3);
rect(:,4) = init_pos(4);
for i = 1 : size(rect,1)
   Y(i) = overlap_ratio(rect(i,:), init_pos);
end
YY = reshape(Y,200,200);

fr_loc = [start_pos(1), start_pos(2)];
to_loc = [start_pos(1)+init_pos(3)-1, start_pos(2)+init_pos(4)-1];

if to_loc(1) > 200
    to_loc(1) = 200;
end
if to_loc(2) > 200
    to_loc(2) = 200;
end
if fr_loc(1) < 1
    fr_loc(1) = 1;
end
if fr_loc(2) < 1
    fr_loc(2) = 1;
end

XX = resp(fr_loc(1):to_loc(1),fr_loc(2):to_loc(2));
YYY = YY(fr_loc(1):to_loc(1),fr_loc(2):to_loc(2));
X_ = reshape(XX, [size(XX,1)*size(XX,2) 1]);
Y_ = reshape(YYY, [size(XX,1)*size(XX,2) 1]);
reg = train_regressor(X_, Y_);

end


function r = overlap_ratio(rect1, rect2) 

inter_area = rectint(rect1,rect2);
union_area = rect1(:,3).*rect1(:,4) + rect2(:,3).*rect2(:,4) - inter_area;

r = inter_area./union_area;
end