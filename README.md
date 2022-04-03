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

You are going to find some really nice charts here soon. 

![Data Example](./sample.gif)


## References and External Links

There are two main data sets that are used to train the model. You can download them using the links below. After you 
have downloaded them, place every sample in a directory called `./data`. The program expects a certain structure of the
data which is as follows:

```
/organ-segmentation
│   .gitignore
│   main.py
│   README.md    
│   requirements.txt    
│
└───/src
|   |   __init__.py
│   │   dataloader.py
|   |   eval.py
│   │   helpers.py
│   │   losses.py
│   │   train.py
│   │   utils.py
│   │   
│   └───/OrganNet25D
│       │   __init__.py
│       │   network.py
│   
└───/data
│   └───/<sample 01>
│   │   └───/<label_folder>
│   │   │   │    BrainStem.nrrd
│   │   │   │    Chiasm.nrrd
│   │   │   │    Mandible.nrrd
│   │   │   │    OpticNerve_L.nrrd
│   │   │   │    OpticNerve_R.nrrd
│   │   │   │    Patroid_L.nrrd
│   │   │   │    Patroid_R.nrrd
│   │   │   │    ...
│   │   │   
│   │   │   <sample>.nrrd
│   │   ...
│
└───/visualizations
```

The variable `label_folder` is by default *structures* but can be adjusted depending on the dataset. `sample 01` is 
named depending on the dataset and `<sample>.nrrd` is also called depending on the dataset (e.g. just img.nrrd). If
the data structure is not as expected, warning and errors will be outputted. You can download the data sets here: 

#### Dataset 1: [Head-Neck Cetuximab collection (46 Samples)](https://www.imagenglab.com/newsite/pddca/ "Dataset 1")
#### Dataset 2: [Martin Valli`eres of the Medical Physics Unit, McGill University, Montreal, Canada (261 samples)](https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT "Dataset 2")


## Acknowledgements

####OrganNet25D
Chen, Z. et al. (2021). A Novel Hybrid Convolutional Neural Network for Accurate Organ Segmentation in 3D Head and Neck 
CT Images. In: , et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. MICCAI 2021. Lecture 
Notes in Computer Science(), vol 12901. Springer, Cham. https://doi.org/10.1007/978-3-030-87193-2_54

####Dataset 1
Raudaschl, P. F., Zaffino, P., Sharp, G. C., Spadea, M. F., Chen, A., Dawant, B. M., … & Jung, F. (2017).
Evaluation of segmentation methods on head and neck CT: Auto‐segmentation challenge 2015.
Medical Physics, 44(5), 2020-2036.

####Dataset 2
McGill University, Montreal, Canada - Special thanks to Martin Vallières of the Medical Physics Unit for providing it