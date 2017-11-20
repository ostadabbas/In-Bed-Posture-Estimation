% s_clsReclinedPCA
%% classify the reclined persons with PCAs. extract the HOG features and extract PCA,
% train it with SVM , save all the features, PCA coeff and trained model to
% specific mat file

%% initialization section
% put right imag folder information. target mat file here.

% clc;clear;
dataRt ='..\dataset\manneSep2';
% dataRt ='..\dataset\humanSep2';
rstImgRt = 'rstImg';
% resAnchorage, Alaskault folder
matFd = 'matData';

% check and create necessary folders
if 7~=exist(rstImgRt)
    mkdir(rstImgRt)
end
if 7~=exist(matFd)
    mkdir(matFd)
end

% set up the specific working folder. <*********************<<<<<<<<
% specificAim='trPoseHumanV2'; % for model traning
specificAim = 'trPoseManneV2';
% specificAim = 'occupied'; % for simple setup

% ***********************>>>>>>>>>>>>
% matNmPrefix = [specificAim,sizeFd];
flgPCA = 1;
if flgPCA
    strFT = 'PCA';
else
    strFT = 'HOG';
end

% ***********************
flgExFts = 1;
flgVisual = 0;  % if visulization is needed or not. Draw different cellsize accuracy later


imgRt = fullfile(dataRt,specificAim);
flgSave = 1; % control if the program will retrain the data.

% cellSize setting ******************
if ~flgTrBoth
    cellDim = 5;   % the edge length
end
cellSize = [cellDim,cellDim];

% PCA steps ******************
flgTrainStep10=1;
% flgTrainStep = 10;
if flgTrainStep10
    stepPCA = 10;
else
    stepPCA =1;
end

% set the default image effect
set(0,'DefaultAxesXGrid','on','DefaultAxesYGrid','on');
set(0,'DefaultLineLineWidth',2); % plot properties
set(0,'DefaultAxesFontSize',15);

%% get the features and lables

imgSets = imageSet(imgRt,'recursive');

% calculate total samples
nSamples  = 0;
for i=1:length(imgSets)
    nSamples = nSamples+  imgSets(i).Count;
end

% nForStep10 = floor(nSamples/10);    % depends on the sample number
% distance 4, the total index numbers
hog = [];
vis = [];

img = read(imgSets(1),1);
[hog, vis] = extractHOGFeatures(img,'CellSize',cellSize);
hogFeaturesSize = length(hog);


if flgExFts
    trainingFeatures=[];
    trainingLabels = [];
    % FOR 1 to sets
    % INIT features numImgx numHOGft
    % FOR j 1 to images
    % ft(j,:)= HOG
    % labels = repmat(setDescription, numImgs, 1);
    % trainft=[trainft; ft];  trainLabels = [
    for classId =1: numel(imgSets)
        numImages = imgSets(classId).Count;
        features= zeros(numImages,hogFeaturesSize,'single');
        for i = 1: numImages;
            img = read(imgSets(classId),i);
            features(i,:)= extractHOGFeatures(img,'CellSize',cellSize);
        end
        
        % if the label is consistent
        labels = repmat(imgSets(classId).Description,numImages,1);
        trainingFeatures = [trainingFeatures;features]; % cascate vertically.
        trainingLabels = [trainingLabels;labels];
    end
    [posePCAcoeff, HOGscores] = pca(trainingFeatures);        % training featues 358*14580
    % Principal component scores are the representations of X in the principal
    % component space. Rows of score correspond to observations, and columns
    % correspond to components.
    posePCAmean = mean(trainingFeatures);
    % the score is 358*357 HOGcoeff 14580*357  X_norm*coeff perhaps only 357
    % coeff, as a random one has only 5 components
    % scores,
    if flgSave % all features no step difference
        save(fullfile(matFd,['fts-',specificAim,'Cel' ,num2str(cellDim),strFT]),'trainingFeatures','trainingLabels','posePCAcoeff','HOGscores','posePCAmean');
        % save reclined64W12S.mat trainingFeatures trainingLabels posePCAcoeff HOGscores posePCAmean
    end
end
%% fit ecoc model and test accuracy, use the principle analysis from 1000 to 14000 features to test the accuracy

% load reclined64W12S
load(fullfile(matFd,['fts-',specificAim,'Cel' ,num2str(cellDim),strFT]));
if flgTrainStep10
     nForStep10 = floor(size(HOGscores,2)/10);
end

% train the model each 10 PCA step. save every model and its accuracy
rclClfLoss = zeros(1,nForStep10);   % initialization
accuracy =[];
poseClfs = {};
poseCVMdls ={};

if flgPCA
    if flgTrainStep10
        for i = 1:nForStep10
            poseClfs{i} = fitcecoc(HOGscores(:,1:i*10),trainingLabels);
            poseCVMdls{i} = crossval(poseClfs{i});
            accuracy(i)=1- kfoldLoss( poseCVMdls{i});
        end
    else
        for i = 1:10
            poseClfs{i} = fitcecoc(HOGscores(:,1:i),trainingLabels);
            poseCVMdls{i} = crossval(poseClfs{i});
            accuracy(i)=1- kfoldLoss( poseCVMdls{i});
        end
    end
    % visualization
    if flgVisual
        figTemp=figure(2);
        
        if flgTrainStep10
            indFts = (1:nForStep10)*10;
        else
            indFts = 1:10;
        end
        
        plot(indFts,accuracy);
        xlabel('Principle component numbers');
        ylabel('Accuracy');
        % title('classification accuracy with different principle compoennt numbers');
        % Convert y-axis values to percentage values by multiplication
        a=[cellstr(num2str(get(gca,'ytick')'*100))];
        % Create a vector of '%' signs
        pct = char(ones(size(a,1),1)*'%');
        % Append the '%' signs after the percentage values
        new_yticks = [char(a),pct];
        % 'Reflect the changes on the plot
        set(gca,'yticklabel',new_yticks);
        set(figTemp,'Units','Inches');
        pos = get(figTemp,'Position');
        set(figTemp, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
        print(figTemp,fullfile(rstImgRt,[specificAim,'Cel' ,num2str(cellDim),strFT,'step',num2str(stepPCA)]),'-dpdf','-r0');
    end
else  % hog feature only not curve only the accuracy of single point
    poseClfs = fitcecoc(trainingFeatures,trainingLabels);
    poseCVMdls = crossval(poseClfs);
    accuracy=1- kfoldLoss( poseCVMdls)
end

if flgSave
    save([matFd,'\',specificAim,'Cel',num2str(cellDim),strFT,'step',num2str(stepPCA)],'poseClfs','poseCVMdls','accuracy','posePCAcoeff','posePCAmean');
end
