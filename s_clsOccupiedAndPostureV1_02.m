% s_clfOccupDataWithTrainedModel
%% prediction evaluation
% initilization
clear;clc;
warning off; 
dataRtTest ='..\dataset\manneSep2';

% mannequin test 
specificAimOccup = 'trOccuManneV3';
specificAimPose = 'trPoseManneV2';

% test folder selection
% human test case
% testFd = 'h23';
% testFd = 'human1Sub2';
testFd = 'tesManneH33Sep2Combined2';
% testFd ='tesHumanV2';

% model step setting*****************
% for mannequin test
occupStep = 10;     % indicate the steps during training
poseStep = 10;
indOccup= 11;
indPose = 11;    % index for the model. 1 for 

% human test
% occupStep=1;
% poseStep=10;
flgSave =1;
flgPCA = 1;
if flgPCA
    strFT = 'PCA';
else
    strFT = 'HOG';
end

% imgRt= fullfile(dataRt,sizeFd,'mannequinOccu');
imgRt = fullfile(dataRtTest,testFd); % test set
imgSets = imageSet(imgRt,'recursive');
totImgNum = 0;  % total image numbers
trueLabels = []; % character array store all the true label, column vec
postFlg =1;  % control the posterior method.  use the fitting ones
flgGau =0; % control the prior model.
flgEnhancedSch =1;

matFd = 'matData';

% cell size 
% cellSize = [5,5];
cellDim = 10;
cellSize =[cellDim,cellDim];

set(0,'DefaultAxesXGrid','on','DefaultAxesYGrid','on');
set(0,'DefaultLineLineWidth',2); % plot properties 
set(0,'DefaultAxesFontSize',15);

%% load model and test
% load training model verion v1.0


load([matFd,'\',specificAimOccup,'Cel',num2str(cellDim),strFT,'step',num2str(occupStep)]);
load([matFd,'\',specificAimPose,'Cel',num2str(cellDim),strFT,'step',num2str(poseStep)]);
% new version modified. Mat into matFd .

trueLabels=[];
for i = 1:length(imgSets)       % l,s case
    totImgNum = totImgNum + imgSets(i).Count;
    
    % choose one true label
    % <<<<<<<<<<<<
    % generate true labels , with fd name
    tempLabelArr = repmat (imgSets(i).Description,imgSets(i).Count,1);
    % generate labels with manditory name instead of imgSets fd names
    %      tempLabelArr = repmat ('pos',imgSets(i).Count,1);
    %>>>>>>>>>>>>>>>>>>>>>    
    trueLabels = [trueLabels; tempLabelArr];
end


% each portion of PCA has different trained models.
cropOccupCoeff = occupPCAcoeff(:,1:indOccup*occupStep); % occupation case
cropPoseCoeff = posePCAcoeff(:,1:indPose*poseStep);
% test 3 categories 100 cases
occupClf = occupClfs{indOccup}; % occupation case
poseClf = poseClfs{indPose};


if 2== postFlg && flgEnhancedSch % only enhanced searching 
    occupClf =  fitSVMPosterior(occupClf);
end

% initialize error and time cost count
errorCntOcc =0;
errorCntPose =0;
% timeCost = zeros(1,300);    % 300 vector
timeCost = zeros(1, totImgNum);
indTot= 1;
pdctPoselabels={};  % prediction labels

clsNum = length(poseClf.ClassNames);
confMat = zeros(clsNum,clsNum); % 3x3 matrix
theta = zeros(1,totImgNum);  %initialize the intial theta, rotate state
% predict only one posture

thetaRang = [-15:5:15, 165:5:195];
% thetaRang = [-15:5:15];
prtOccu = {};
for i = 1:length(imgSets)       % l,s case
    indCls = strmatch(imgSets(i).Description,poseClf.ClassNames);   % which class
    for j = 1:imgSets(i).Count
        img = read(imgSets(i),j);
        %         imshow(img)
        % occupation test
        tic        

        if flgEnhancedSch
            % mu the mean of the shift and angle default [0,0]
            % std the standard deviation of the  shift and angle default [3, 20]
            % default parameters
            [Iout, occupLabel,occuScore, Xtrans, theta(indTot),hog,vis] = SearchStateSpace(img,occupPCAmean,...
                cropOccupCoeff,occupClf,0,cellSize,-8:4:8,thetaRang,postFlg,[0,0],[3,20],flgGau); % default -20:20:20 thing
            labels = occupLabel;
            %                       [hog_4x4,vis4x4]= extractHOGFeatures(Iout,'CellSize',cellSize);
        else
            [hog,vis]= extractHOGFeatures(img,'CellSize',cellSize);
            ftsPCA = ToPCAspace(hog,occupPCAmean,cropOccupCoeff);
            labels = predict(occupClf,ftsPCA);
        end
        %         timeCost((i-1)*100+j)=toc;
        %  'pos'=='pos' gives [1 1 1]. Compare each single character
        if ~strcmp(labels,'pos')
            errorCntOcc=errorCntOcc+1;    % not positive
        end
        
        prtOccu{indTot}= labels;
        
        
        poseFtsPCA  = ToPCAspace(hog,posePCAmean,cropPoseCoeff);
        tmpPdct= predict(poseClf,poseFtsPCA);
        % updata confMat
        indPdct = strmatch(tmpPdct, poseClf.ClassNames);
        confMat(indCls, indPdct)=confMat(indCls, indPdct)+1;    % the certain part +1;
        
        pdctPoselabels(indTot)= cellstr(tmpPdct);
        %         timeCost((i-1)*100+j)=toc;
        timeCost(indTot) = toc;
        indTot = indTot +1; % total index
        %         if ~strcmp(labels,imgSets(i).Description)
        %             errorCntPose=errorCntPose+1;
        %         end
    end
end

% compare result
if flgGau 
    display('gaussian prior');
else 
    display('uniform prior');
end
if 1==postFlg
    display('sigmoid posterior');
else
    display('linear score');
end
    
trueLabelsCell = cellstr(trueLabels);
CPPos = classperf(cellstr(trueLabels), pdctPoselabels);
fprintf('posture classification accuracy is %4d',CPPos.CorrectRate);
% CPPos.CorrectRate
confmat=confusionmat(cellstr(trueLabels),pdctPoselabels);  % known and predict
confmat
% display('total occupation test error');
fprintf('total occupation test error is %4d',double(errorCntOcc)/(indTot-1));


if flgSave
    save(fullfile(matFd,['rst-',testFd,'Cel',num2str(cellDim),strFT,'step',num2str(poseStep),'Ind',num2str(indPose),'Enhc',num2str(flgEnhancedSch)]),...
    'confmat','CPPos');
    % named after pose step 
end
