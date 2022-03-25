# Organ Segmentation in 3D Head and Neck CT Images

## Getting Started

**This is what you gotta do to get started:**

1) This cool stuff
2) That cool stuff as well
3) And mainly `training --all` the thing
4) Using the `inference --deep` mode as well
5) Have great results


## Deep Learning in Organ Segmentation

#### A Novel Hybrid Convolutional Neural Network for Accurate Organ Segmentation  in 3D Head and Neck CT Images

*Radiation therapy (RT)* is widely employed in the clinic for the treatment of head and neck (HaN) cancers. An essential 
step of RT planning is the accurate segmentation of various organs-at-risks (OARs) in HaN CT images. Nevertheless, 
segmenting OARs manually is time- consuming, tedious, and error-prone considering that typical HaN CT images contain 
tens to hundreds of slices. Automated segmentation algo- rithms are urgently required. Recently, convolutional neural 
networks (CNNs) have been extensively investigated on this task. Particularly, 3D CNNs are frequently adopted to process 
3D HaN CT images. There are two issues with na ̈ıve 3D CNNs. First, the depth resolution of 3D CT images is usually 
several times lower than the in-plane resolution. Direct employment of 3D CNNs without distinguishing this difference 
can lead to the extraction of distorted image features and influence the final segmentation performance. Second, a 
severe class imbalance prob- lem exists, and large organs can be orders of times larger than small organs. It is 
difficult to simultaneously achieve accurate segmentation for all the organs. To address these issues, we propose a 
novel hybrid CNN that fuses 2D and 3D convolutions to combat the different spa- tial resolutions and extract effective 
edge and semantic features from 3D HaN CT images. To accommodate large and small organs, our final model, named 
OrganNet2.5D, consists of only two instead of the clas- sic four downsampling operations, and hybrid dilated 
convolutions are introduced to maintain the respective field. Experiments on the MIC- CAI 2015 challenge dataset
demonstrate that OrganNet2.5D achieves promising performance compared to state-of-the-art methods.


## Results and what you get out of this

You are going to find some really nice charts here soon


## References and External Links

Data can be downloaded here `https://www.imagenglab.com/newsite/pddca/`. Place the data in a directory called `./data`, 
which is excluded from the GIT Repository due to size limitations.


## Acknowledgements

Zijie Chen - Shenzhen Key Lab of Computer Vision and Pattern Recognition, China

Cheng Li - Shanghai AI Lab, Shanghai, China

Junjun He - Shenzhen Yino Intelligence Techonology Co., Ltd., Shenzhen, Guangdong, China

Jin Ye - Shenying Medical Technology (Shenzhen) Co., Ltd., Shenzhen, Guangdong, China

Diping Song - Paul C. Lauterbur Research Center for Biomedical Imaging, China

Shanshan Wang - School of Biomedical Engineering, Shanghai Jiao Tong University, Shanghai, China

Lixu Gu - Institute of Medical Robotics, Shanghai Jiao Tong University, Shanghai, China

Yu Qiao - Peng Cheng Laboratory, Shenzhen, Guangdong, China