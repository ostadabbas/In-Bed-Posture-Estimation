# in-bed posture estimation 
Shuangjun Liu,  "In-bed posture estimation" ,  Assistive Computer Vision and Robotics (ACVR) 2017 

contact: Sarah Ostaddabas (ostadabbas@ece.neu.edu )
Shuangjun Liu (shuliu@ece.neu.edu)

## project description
This project aims at classifying in-bed posture from off the shelf low cost webcam with potential applications in health care and sleeping research. 

## before everything 
- Install matlab2016a.(Higher version can work as long as they keep the same toolbox APIs. )
Make sure machine learning toolbox, computer vision toolbox and Bioinformatics toolbox are installed.
- download manne2 dataset from our [website](http://www.northeastern.edu/ostadabbas/2017/11/19/in-bed-general-posture-estimation/)
- Put code and dataset folder in the same folder. Unzip the manne2 into dataset folder. You can also specify the dataset path in code.


## Training
Run script s_trPosOccu.m. It will generate a group of occupation and posture classification models with different HOG cellsize. 
You can customize the cellsize as you need. 10 gives optimal performance in our test. So you can simply train model only with cellsize 10.
Result will be saved under matData folder. 

## Testing 
s_clsOccupiedAndPostureV1_02.m to evaluate on the test set. 

s_drawCellAccuracy.m visulizes the cross validation result of each model. 

## Citation
Please cite our paper if it helps your research:

	@inproceedings{liu2017vision,
	  title={A Vision-Based System for In-Bed Posture Tracking},
	  author={Liu, Shuangjun and Ostadabbas, Sarah},
	  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  pages={1373--1382},
	  year={2017}
	}



