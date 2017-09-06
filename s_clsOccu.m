% s_clfOccu
%% prediction evaluation for occupation only  
% initilization
clear;clc;
warning off; 
% dataRtMdl ='..\dataset\manneSep2';
% dataRtTest = '..\dataset\humanSep2';
dataRtTest = '..\dataset\manneSep2';
% choose one for test
% for mannequin mixed
% specificAim='wkMannequinV1';
% testFd = 'wkMannequinV1'

% for smaller size people
% specificAim = 'wkMannequinV1';
% for classifier identifier
% specificAimOccup='wkOccuSimV1'; % for mat identifier, occupation test

% load trained model 
% human test
% specificAimOccup='wkOccuHumSub1';
% specificAimPose='human1Sub1'; % poseture case test

% mannequin test 
specificAimOccup = 'trOccuManneV3';
% specificAimPose = 'trPoseHumanV2';

% test folder selection
% for lower height of the bed
% testFd ='variational\height23'; % image fd indicator subsub folder case
% for rotation case
% testFd ='variational\rot14.6'; % image fd indicator subsub folder case
% h33 case
% human test case
% testFd = 'human1';
% testFd = '\mannequin2';
% testFd = 'h33'
% testFd= 'h33v1'
% testFd ='manneR180';
% testFd = 'h33R180';
% testFd ='h33Combined'
% testFd = 'h23';
% testFd = 'human1Sub2';
% testFd = 'tesManneH33Sep2Combined2';
% testFd ='tesHumanV2';
testFd = 'tesOccuManneV3';

% figure setting
set(0,'DefaultAxesXGrid','on','DefaultAxesYGrid','on');
set(0,'DefaultLineLineWidth',2); % plot properties 
set(0,'DefaultAxesFontSize',15);

% fd settings
rstImgRt = 'rstImg';
% result folder
matFd = 'matData';

% check and create necessary folders
if 7~=exist(rstImgRt)
    mkdir(rstImgRt)
end
if 7~=exist(matFd)
    mkdir(matFd)
end

% model step setting*****************
% for mannequin test
occupStep = 10;     % indicate the steps during training
poseStep = 10;
indOccup= 5;
indPose = 5;    % index for the model. 1 for 


% human test
% occupStep=1;
% poseStep=10;
flgSave =1;
flgPCA = 1;
flgVis = 1;     % if visualization
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
% load rclClfPCAv1
% load rclClfPCAv1

% version v1.1
% load(['clfier-',matNmPrefix]);  % only the reclined test this time

% the ecoc model version
% load(['occupClfPCA_STEP1-',specificAimOccup,sizeFd]);
% load SVM model
% step 10 model
% original working load 
% load(['occupClfPCA_STEP10V1_01-',specificAimOccup,sizeFd]);
% load(['clfierStp10-',specificAimPose,sizeFd]);

load([matFd,'\',specificAimOccup,'Cel',num2str(cellDim),strFT,'step',num2str(occupStep)]);
% load([matFd,'\',specificAimPose,'Cel',num2str(cellDim),strFT,'step',num2str(poseStep)]);
% new version modified. Mat into matFd .
% load([matFd,'/','occupClfPCA_STEP1V1_01','Cel',num2str(cellSize(1)),'-',specificAimOccup,sizeFd]);
% load([matFd,'/','clfierStp10','Cel',num2str(cellSize(1)),'-',specificAimPose,sizeFd])

% load([matFd,'\','clfierStp10','Cel',num2str(cellSize(1)),'-',matNmPrefix])
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


% hog features
% cellSize=[4,4 ];

% occupation model
% step1 7th model
% indOccup = 7;   %
% step10 4th model


% each portion of PCA has different trained models.
cropOccupCoeff = occupPCAcoeff(:,1:indOccup*occupStep); % occupation case
% cropPoseCoeff = posePCAcoeff(:,1:indPose*poseStep);
% test 3 categories 100 cases
occupClf = occupClfs{indOccup}; % occupation case
% poseClf = poseClfs{indPose};


if 2== postFlg && flgEnhancedSch % only enhanced searching 
    occupClf =  fitSVMPosterior(occupClf);  % if postFlg 2, fit the poseterior 
end

% initialize error and time cost count
errorCntOcc =0;
errorCntPose =0;
% timeCost = zeros(1,300);    % 300 vector
timeCost = zeros(1, totImgNum);
indTot= 1;
pdctPoselabels={};  % prediction labels

theta = zeros(1,totImgNum);  %initialize the intial theta, rotate state
thetaRang = [-15:5:15, 165:5:195];
prtOccuLabels  = {};
scoresOccu = [];
for i = 1:length(imgSets)       % l,s case
%     indCls = strmatch(imgSets(i).Description,poseClf.ClassNames);   % which class
    for j = 1:imgSets(i).Count
        img = read(imgSets(i),j);
        tic        

        if flgEnhancedSch
            % mu the mean of the shift and angle default [0,0]
            % std the standard deviation of the  shift and angle default [3, 20]
            % default parameters
            [Iout, occupLabel,scoreOccu, Xtrans, theta(indTot),hog,vis] = SearchStateSpace(img,occupPCAmean,...
                cropOccupCoeff,occupClf,0,cellSize,-8:4:8,thetaRang,postFlg,[0,0],[3,20],flgGau); % default -20:20:20 thing
            labels = occupLabel;
            %                       [hog_4x4,vis4x4]= extractHOGFeatures(Iout,'CellSize',cellSize);
        else
            [hog,vis]= extractHOGFeatures(img,'CellSize',cellSize);
            ftsPCA = ToPCAspace(hog,occupPCAmean,cropOccupCoeff);
            [labels,scoreTemp] = predict(occupClf,ftsPCA);
            scoreOccu = scoreTemp(2);   % only the positive score -1 to 1 
        end
        prtOccuLabels{indTot}= labels;      % updata prediction 
        scoresOccu(indTot)= scoreOccu;
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
% CPPos = classperf(cellstr(trueLabels), pdctPoselabels);
CPPos = classperf(cellstr(trueLabels),prtOccuLabels);
display('posture classification');
CPPos.CorrectRate
% confmat=confusionmat(cellstr(trueLabels),pdctPoselabels);  % known and predict
confmat = confusionmat(cellstr(trueLabels),prtOccuLabels)
% calculate the ROC curve 
[X,Y,T,AUC  ]=  perfcurve(cellstr(trueLabels),scoresOccu,'pos');
% display('AUC:');
% AUC
fprintf('AUC is %d',AUC);


if flgSave
    save(fullfile(matFd,['rst-',testFd,'Cel',num2str(cellDim),strFT,'step',num2str(poseStep)]),...
    'confmat','CPPos','X','Y','T','AUC');
    % named after pose step 
end
if flgVis
    figTemp= figure(3);
    plot(X,Y,'LineStyle','-','LineWidth',3);
    xlabel('False positive rate');
    ylabel('True positive rate')  ;
    %     title('ROC for Classification by Logistic Regression')
    
    %    print figure
    set(figTemp,'Units','Inches');
    pos = get(figTemp,'Position');
    set(figTemp, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
%     print(figTemp,'occupStep10','-dpdf','-r0');
    print(figTemp,fullfile(rstImgRt,['ROC-',testFd,'Cel' ,num2str(cellDim),strFT,'step',num2str(occupStep)]),'-dpdf','-r0');

end