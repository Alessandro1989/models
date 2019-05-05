%try to convert in csv
%load('C:\Users\YourUserName\Downloads\mnist_all.mat');


%load('C:\tmp\svhn_data\train\digitStruct.mat');
load('digitStruct.mat');

%try
%csvwrite('digitStruct.csv',digitStruct(0));

%for i = 1:length(digitStruct)
fprintf('running see bboxes' );

fileID = fopen('digitStruct_eval.txt','w');


%for i = 1:10
for i = 1:length(digitStruct)
    %im = imread([digitStruct(i).name]);
    for j = 1:length(digitStruct(i).bbox)
        fprintf(fileID, 'name: %s;',digitStruct(i).name);

        %[height, width] = size(im);
        %aa = max(digitStruct(i).bbox(j).top+1,1);
        %bb = min(digitStruct(i).bbox(j).top+digitStruct(i).bbox(j).height, height);
        %cc = max(digitStruct(i).bbox(j).left+1,1);
        %dd = min(digitStruct(i).bbox(j).left+digitStruct(i).bbox(j).width, width);
        
        %imshow(im(aa:bb, cc:dd, :));
        fprintf(fileID, 'label: %d;',digitStruct(i).bbox(j).label );
        fprintf(fileID, 'top: %d;',digitStruct(i).bbox(j).top );
        fprintf(fileID, 'left: %d;',digitStruct(i).bbox(j).left );
        fprintf(fileID, 'width: %d;',digitStruct(i).bbox(j).width );
        fprintf(fileID, 'height: %d;',digitStruct(i).bbox(j).height );
        fprintf(fileID, '\n');
        %pause;
    end
end
fprintf("done");
fclose(fileID);
