# SSD: Single Shot MultiBox Detector
``` 
├── train.py: Using VGG backbone as the backbone of the SSD network for training  
└── test.py: Use the COCO indicator of the trained weight validation/test data and generate the record_mAP.txt file
```
## Data set, using PASCAL VOC2012 and BDD100K data set (download and put it in the current folder of the project)
* Pascal VOC2012 train/val dataset download address：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* BDD100K：https://bdd-data.berkeley.edu/portal.html#download #Download 100K images and Detection 2020 Labels


## training method
* Make sure to prepare the dataset ahead of time
* Modify the custom data set and convert the corresponding file in the preprocess folder to a voc format data set
* Change the custom dataset name to VOC2007 or VOC2012
* Use the corresponding backbone

## pre_trained weights

* download from :https://github.com/psxzz13/SSD
* pre trained weights download from : https://drive.google.com/drive/folders/15YU9zNuSYd_AfXQSVZGAW68OaNAohUlz?usp=sharing
* vgg16_reducedfc: for SSD_VGG
* nvidia_ssdpyt_fp32: for SSD_chaned_detection
* SSD_changed_detection: Before trainnning, download the weights and copy that to SSD_changed_detection/src path
* SSD_VGG: Before trainnning, download the weights and copy that to SSD_VGG_Zixuan zhang/weights path
