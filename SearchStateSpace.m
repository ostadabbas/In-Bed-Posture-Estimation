function [Iout, labelMax, scoreMax, Xtrans, theta,hogOut,visOut] = SearchStateSpace(I,PCAmean,...
    PCAcoeff,occupClf, drawFlg,cellSize, xGrid, thetaGrid,postFlg,mu, std, flgGau,flgSimMap)

% function [Iout, label, Xtrans, theta] = SearchStateSpace(I,  PCAmean,...
% PCAcoeff, occupClf, cellSize, xGrid, thetaGrid, drawFlg) 
% gets images and searches nearby space to find the highest response. 
% The searching space is x: -8 to 8 with step 4, and rotation -20:20 
% with step 20. It returns the highest score response image and 
% corresponding label, and the states of the detection window
% ( X_trans and rotate here), 
% this function only works for the Xtrans only with a standard 64 width 
% image. This function is a simplified edition specialized for our 
% detection purpose with fixed image width, pre trained classifier and 
% PCA parameters.  This could be improved to be a more flexible one 
% like taking the search space parameters. Here we don’t give check 
% mechanism for the PCA and classifier, you should check them before input 
% into this function. 
% 
% output: Iout, the output image. 
% label, the classification result. 
% Xtrans, the translation state in x direction
% theta, the rotate state
% hogOut, visOut = the hog result

% input: Iw64, the image with 64 width. 
% PCAmean, the PCA space mean value
% PCAcoeff, the coefficient of the PCA space, 
% occupClf, the pre-trained classifier
% drawFlg,  the draw flag to indicate if it will draw the response pictures. 
% cellSize, the HOG feature cellsize default [4,4]
% xGrid, the x searching space 
% thetaGrid, the rotation searching space. 
% drawFlg, it draw the image in a large number figure 
% postFlg,posterior model, then a kflodPredict method will be employed, and
% normailized with respect to the grid numbers, 1 for linear mapping 2, for
% posterior method fitting with build-in function  
% mu the mean of the shift and angle default [0,0] 
% std the standard deviation of the  shift and angle default [3, 20]
% flgGau if the gaussian is used. 
% flgSimMap if employ matlab buid-in simple mapping method. 

% history: 
% 17/8/14, return the maxScores back for ROC plot 

if nargin<4
    error('4 parameters are needed');
end

switch nargin   % setup default parameters 
    case 4
        drawFlg = 0;
        cellSize = [4,4];
        xGrid = -8:4:8;
                thetaGrid = -15:5:15;
        postFlg=0;
        mu=[0,0];
        std = [3,15];
        flgGau=1; 
        flgSimMap=0;
    case 5
       cellSize = [4,4];
        xGrid = -8:4:8;
        thetaGrid = -15:5:15;
        postFlg=0;        
        mu=[0,0];
        std = [3,15];
        flgGau=1; 
        flgSimMap=0;
    case 6
        xGrid = -8:4:8;
        thetaGrid = -15:5:15;
        postFlg=0;
        mu=[0,0];
        std = [3,15];
        flgGau=1; 
    case 7
        thetaGrid = -15:5:15;
         postFlg=0;
         mu=[0,0];
        std = [3,15];
        flgGau=1; 
        flgSimMap=0;
    case 8 
        postFlg=0;
        mu=[0,0];
        std = [3,15];
        flgGau=1; 
        flgSimMap=0;
    case 9
        mu=[0,0];
        std = [3,15];
        flgGau=1; 
        flgSimMap=0;
    case 10
        std = [3,15];
        flgGau=1; 
        flgSimMap=0;
    case 11
        flgGau=1;
        flgSimMap=0;
    case 12
        flgSimMap=0;
end
xL = length(xGrid);
thetaL = length(thetaGrid);
normFac = 1.0/(xL*thetaL);
scores = zeros(thetaL, xL);
scoreMax = -100;    % default lowest score, any response should be higher than this one
labelMax = 'neg';   % negative class in our case, if you want to use this method in your 
% pipeline, you should use the pos and neg label as this one. 
Xtrans = 0;
theta = 0; 
iMax= 0;
jMax = 0;

% no need to processed 
% if 2==postFlg   % if a sigmoid function 
%     occupClf= fitSVMPosterior(occupClf);    % change to a posterior model 
% end
% prior generation of weights 
if flgGau 
        wtSft = normpdf(xGrid,mu(1),std(1));
        
        % one gaussian or 2 gaussians 
        if max(thetaGrid)- min(thetaGrid)>90
            wtRot = (normpdf(thetaGrid,mu(2),std(2))+normpdf(thetaGrid,mu(2)+180,std(2)))/2;   
        else
            wtRot = normpdf(thetaGrid,mu(2),std(2));
        end
        % rotation is symmetric version. 
        wtMat = wtRot'*wtSft;
else
    wtMat = ones(length(thetaGrid), length(xGrid))* normFac;
end

for i =1: length(thetaGrid)
    for j = 1:length(xGrid)
        if flgSimMap
            Itr = imrotate(I,-thetaGrid(i),'bilinear','crop');
            Itr = imtranslate(Itr,[-xGrid(j),0]);
        else
            Itr = TransformImg(I,[-xGrid(j),0],-thetaGrid(i));
        end
        [hog_4x4, vis4x4]= extractHOGFeatures(Itr,'CellSize', cellSize);
         occupFtsPCA = ToPCAspace(hog_4x4,PCAmean,PCAcoeff);
            [label,score]= predict(occupClf,occupFtsPCA);   % 2 score from here
            scores(i,j)=score(2)*wtMat(i,j);   % get possitive class score only
%             labels{k,j}=label; 
            if scores(i,j)>scoreMax
                 scoreMax = scores(i,j);
                 iMax = i;
                 jMax = j;
                labelMax = label;
                Xtrans = xGrid(j);
                theta = thetaGrid(i);
                Iout =Itr;
                hogOut = hog_4x4;
                visOut = vis4x4;
            end
    end
end
fontSize= 16;
% if 1, simple mapping processing, if 2 then it's already prior
if 1==postFlg   % if a sinple mapping  
   scores= (scores+1)/2;
   scoreMax = (scoreMax+1)/2;
end

% this is for the svm directly 
% if postFlg
%     scoreMax = (scoreMax+1)/2*normFac;
%     scores = (scores+1)/2*normFac;
% end

% this is for the posterior model 


if drawFlg 
    [X,THETA]= meshgrid(xGrid,thetaGrid);
    figure(200);    
	surf(X,THETA, scores);
    set(gca,'FontSize',fontSize);
    xlabel('X shift (pixel)','FontSize',fontSize);
    ylabel('Rotation (degree)','FontSize',fontSize);
    zlabel('Response scores','FontSize',fontSize);
end
        
    

        