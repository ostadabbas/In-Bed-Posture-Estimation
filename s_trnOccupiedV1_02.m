% sTrnClfOccupiedV1_01
% orignal use ecoc mechanism, here use the
% train the data with 2 pos and neg class for bed occupation detection
% clear;clc;

%% initialization of the dataset parameters (folder ,etc?
% imgRoot ='..\reclined data\OccupationCatagory\W64';
% imgRoot ='..\dataset\manneSep2';
% imgRoot = '..\dataset\humanSep2';
imgRoot = '..\dataset\manneSep2';

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

% sizeFd = '64W'; % no need to label
% name:  datafd, cell{N}, FT{PCA or HOG}, STEP{1,10}
% datafd should show the purpose trOccu for the occupation detection

% set up the specific working folder. <*********************<<<<<<<<

% for simlab setup mannequin
specificAim='trOccuManneV3'; % for model traning

% % for simple setup not right that is for posture
% specificAim = 'occupied';

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
flgVisual = 0;  % if visulization is needed or not

imgRt = fullfile(imgRoot,specificAim);
flgSave = 1; % control if the program will retrain the data.

% cellSize setting ******************
if ~exist('flgTrBoth') % if not assigned then assign cellDim
    cellDim = 10;   % the edge length
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


% default plot drawing
set(0,'DefaultAxesXGrid','on','DefaultAxesYGrid','on');
set(0,'DefaultLineLineWidth',2); % plot properties
set(0,'DefaultAxesFontSize',12);
%% data recruitment and processing, key parameter acquired
imgSets = imageSet(imgRt,'recursive');
% negSet = imageSet(negRoot);

% get sample number
nSamples  = 0;
for i=1:length(imgSets)
    nSamples = nSamples+  imgSets(i).Count;
end

% the total index numbers
% nForStep10 = floor(nSamples/10);    % should comes from teh hog scores columns more accurate

img = read(imgSets(1),1);
hog = [];
vis = [];

[hog, vis] = extractHOGFeatures(img,'CellSize',cellSize);
hogFeaturesSize = length(hog);


%% train PCA features
if flgExFts
    trainingFeatures=[];
    trainingLabels = [];
    
    for classId =1: numel(imgSets)      % two sets, pos and neg
        numImages = imgSets(classId).Count;
        features= zeros(numImages,hogFeaturesSize,'single');
        for i = 1: numImages;
            img = read(imgSets(classId),i);
            features(i,:)= extractHOGFeatures(img,'CellSize',cellSize);
        end
        labels = repmat(imgSets(classId).Description,numImages,1);
        trainingFeatures = [trainingFeatures;features]; % cascate vertically.
        trainingLabels = [trainingLabels;labels];
    end
    
    [occupPCAcoeff, HOGscores] = pca(trainingFeatures);

    occupPCAmean = mean(trainingFeatures);   % this is for future detection,
    
    
    if flgSave
        % save occupationFts.mat trainingFeatures trainingLabels occupPCAcoeff HOGscores occupPCAmean
        save([matFd,'\','fts-',specificAim,'Cel' ,num2str(cellDim),strFT],'trainingFeatures','trainingLabels','occupPCAcoeff','HOGscores','occupPCAmean');
        % HOGscores is the result after PCA coeff transformation
    end
end


%% train binary classifier
% load occupationFts
load([matFd,'\','fts-',specificAim,'Cel' ,num2str(cellDim),strFT]);

if flgTrainStep10
    nForStep10 = floor(size(HOGscores,2)/10);
end

% initialization
occupClfs={};
occupCVMdls={};

if flgTrainStep10 % for number is different so if out of for
    occupAccur = zeros(1,nForStep10);   % initialization
    for i = 1:nForStep10
        % kernel one
%         SVMModel = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF',...
%     'KernelScale','auto');

        occupClfs{i} = fitcsvm(HOGscores(:,1:i*10),trainingLabels); % linear kernel 
        occupCVMdls{i} = crossval(occupClfs{i});
        occupAccur(i)=1- kfoldLoss( occupCVMdls{i});
    end
else
    for i = 1:10
        %         occupClfs{i} = fitcecoc(HOGscores(:,1:i),trainingLabels);
        occupClfs{i}= fitcsvm(HOGscores(:,1:i),trainingLabels);
        occupCVMdls{i} = crossval(occupClfs{i});
        occupAccur(i)=1- kfoldLoss( occupCVMdls{i});
    end
end
if flgSave
    %       save([matFd,'\occupClfPCA_STEP10V1_01','Cel',num2str(cellSize(1)),'-',matNmPrefix],'occupClfs','occupCVMdls','occupAccur','occupPCAcoeff','occupPCAmean');
    save([matFd,'\',specificAim,'Cel',num2str(cellDim),strFT,'step',num2str(stepPCA)],'occupClfs','occupCVMdls','occupAccur','occupPCAcoeff','occupPCAmean');
end
% visulization
if flgVisual
    
    if flgTrainStep10
        indFts = (1:nForStep10)*10;
    else
        indFts = 1:10;
    end
    
    figTemp = figure(1);
    plot(indFts,occupAccur);
    xlabel('Principle component numbers');
    ylabel('Accuracy');
    % Convert y-axis values to percentage values by multiplication
    a=[cellstr(num2str(get(gca,'ytick')'*100))];
    % Create a vector of '%' signs
    pct = char(ones(size(a,1),1)*'%');
    % Append the '%' signs after the percentage values
    new_yticks = [char(a),pct];
    % 'Reflect the changes on the plot
    set(gca,'yticklabel',new_yticks);
    grid on;
    
    set(figTemp,'Units','Inches');
    pos = get(figTemp,'Position');
    set(figTemp, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
%     print(figTemp,'occupStep10','-dpdf','-r0');
    print(figTemp,fullfile(rstImgRt,[specificAim,'Cel' ,num2str(cellDim),strFT,'step',num2str(stepPCA)]),'-dpdf','-r0');
end
