% s_drawCellAccuracy
% laod the models and evaluate the accuracy performance. Models need to be
% trained before hand.

clc;clear;
% initialize the figure properties
set(0,'DefaultAxesXGrid','on','DefaultAxesYGrid','on');
set(0,'DefaultLineLineWidth',2); % plot properties
set(0,'DefaultAxesFontSize',15);
set(0,'DefaultTextFontSize',15);

% flag and settings
cellDimArr = 5:5:30;
flgSave = 1;

% fd location
matFd = 'matData';
rstImgRt = 'rstImg';
% check and create necessary folders
if 7~=exist(rstImgRt)
    mkdir(rstImgRt)
end
if 7~=exist(matFd)
    mkdir(matFd)
end

specificAim = 'trOccuManneV3';  % model to evaluate, occu or pose
strFT = 'PCA';  % choose PCA or HOG,
stepPCA = 10;

% setup figuer
figTemp = figure(100); clf; hold on;

% initialize the accuracy array
accur = {};
% legend names
legendNms = {};
% initialize the bound
accurMax = 0;
accurMin = 1;

for i = 1:length(cellDimArr)
    cellDim = cellDimArr(i);
    % load model
    models = load([matFd,'\',specificAim,'Cel',num2str(cellDim),strFT,'step',num2str(stepPCA)]);
    if findstr(specificAim,'Occu')
        for j = 1:length(models.occupCVMdls)
            accur{i}(j) = 1 - kfoldLoss(models.occupCVMdls{j});
        end
    else
        for j = 1:length(models.poseCVMdls)
            accur{i}(j) = 1 - kfoldLoss(models.poseCVMdls{j});
        end
    end
    accurMax = max(max(accur{i}),accurMax);
    accurMin = min(min(accur{i}),accurMin);
    
    legendNms{i}=['cellsize',num2str(cellDim)];
    % visualize save figures
    plot((1:j)*stepPCA,accur{i}*100,'LineStyle','-','LineWidth',3);
end

legend(legendNms,'Location','SouthEast','FontSize',12);
xlabel('PCA numbers');
ylabel('Accuracy,%');

if flgSave
    rstNm = fullfile(rstImgRt,['accCell-',specificAim,strFT, 'step',num2str(stepPCA)]);
    set(figTemp,'Units','Inches');
    pos = get(figTemp,'Position');
    set(figTemp, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
    print(figTemp,rstNm,'-dpdf','-r0');
end
