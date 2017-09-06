# in-bed posture estimation 
Shuangjun Liu,  "[In-bed posture estimation](www.website coming later)" ,  Assistive Computer Vision and Robotics (ACVR) 2017 

contact: metero616@gmail.com

## project description
This project aims at classifying in-bed posture from off the shelf low cost webcam with potential applications in health care and sleeping research. 

## before everything 
- Install matlab2016a with machine learning toolbox. (Higher version can work as long as they keep the same toolbox APIs. )
- rename the dataset folder to dataset 
- put the dataset folder and code folder into the same directory

## Training
Run script s_trPosOccu.m. It will generate a group of occupation and posture classification models with different HOG cellsize. 

## Testing 
s_drawCellAccuracy.m gives accuracy with different PCA components against different cell size. 

s_clsOccupiedAndPostureV1_02.m test the accuracy for both occupation and postures classifications. 




