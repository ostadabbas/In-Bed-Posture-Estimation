% s_trPosOccu
% traint the pose and occupation together, change the cell size in an array
% and train the model generate result image and save the model data 
clear;clc;
flgTrBoth = 1; 
cellDimArr  = 5:5:30;  % assign all the possible cell size 

for ind_cell =1:length(cellDimArr)
    cellDim = cellDimArr(ind_cell);
    s_trnOccupiedV1_02;
    s_trnPoseV1_02;   % commment for occupation only training 
end
    

flgTrBoth = 0;  % clear in case used seperately later 