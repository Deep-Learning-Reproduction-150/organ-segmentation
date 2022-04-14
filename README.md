# Organ Segmentation in 3D Head and Neck CT Images

![Data Example](./sample.gif)
Example of the data to work with from [1]

## Getting Started

This program is based on a *Runner logic*. You create a json file that contains the 
description of a *job*, the Runner will load it and execute whatever you specified, 
i.e. what dataset to use, what transformations to apply, what model to train and so on.
There is a sample configuration file located at `./config_sample.json` that contains the
setup you need to reproduce this papers results (see below). To get started, follow the
steps mentioned below.

**Follow those four steps to get started:**

1) Run `pip install -r requirements.txt` to install all required packages
2) Copy the sample config `./config_sample.json` to `./configs` (name it whatever)
3) Open the `main.py` file and add the path to your config `/config/<your-config>` 
4) Run the main file `python main.py`

```python
from src.Runner.Runner import Runner


# Add all the jobs, that you want to run, here
jobs = ['config/<your-config>.json']

# Main guard for multithreading the runner "below"
if __name__ == '__main__':

    # Create a runner instance and pass it the jobs
    worker = Runner(jobs=jobs, debug=True)

    # Start working on the jobs until all are finished
    worker.run()
```

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

*Insights in the results will be provided as soon as we have them* 

## References and External Links

There are two main data sets that are used to train the model. You can download them using the links below. After you 
have downloaded them, place every sample in a directory called `./data`. We used the transformed dataset created by the great 
[Prerak Mody](https://github.com/prerakmody "GitHub of Prerak Mody"), where the voxel sizes have been adjusted in all 
scans to have the same dimensions. 

#### Dataset (original): [Head-Neck Cetuximab collection (46 Samples)](https://www.imagenglab.com/newsite/pddca/ "Dataset")
#### Dataset (transformed): [Transformed Dataset from Prerak Mody](https://github.com/prerakmody/hansegmentation-uncertainty-qa/releases "Dataset Resampled")

```
/organ-segmentation
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
└───/src
|   |   ...
|   
│   .gitignore
│   main.py
│   README.md    
│   requirements.txt
│   sample_config.json    
│   sample.gif       
```

## Further Information: Job Configuration

The configuration file specifies what the Runner should do. You can also add your own 
components e.g. `Loss Functions`, `Trasforms` or other `Models` in the dedicated locations.
Transforms can be also choosen from `torchvision.transform`. The runner first looks for a
matching transform in `src/Data/transforms.py`, if there is nothing with the specified name, 
it will try to import it from torchvision. If you don't choose dataset labels, the dataset
will specify an own order of labels (that will be the channels of the tensors). 

```javascript
config = {
    "name": "<name>",
    // Whether the runner should recover a model from a checkpoint
    "resume": true, 
    // Whether data shall be preloaded to speed up training (can be RAM-intensive) 
    "preload": true,
    // Weights and Biases setup
    "wandb_project_name": "<wandb project>",
    // Sample prediction slices to see current behavior logged 
    "wandb_prediction_examples": 8,
    "wandb_api_key": "<your wandb api key>",
    // Model specifications
    "model": {
        "name": "OrganNet25D",
        // Add parameters to the model here
    },
    "training": {
        "epochs": 300,
        "detect_bad_gradients": false,
        "grad_norm_clip": 1,
        // Split ratio between training and evaluation dataset
        "split_ratio": 0.77,
        "batch_size": 2,
        // Specification of the loss function 
        "loss": {
            "name": "CombinedLoss",
            // Add parameters to the loss function here
        },
        // Specification of the optimizer
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.001,
            // Add more parameters here
        },
        // Specification of the learning rate scheduler
        "lr_scheduler": {
            "name": "MultiStepLR",
            "gamma": 0.1,
            "milestones": [50, 100]
        },
        // Specification of the data set
        "dataset": {
            "root": "./data/train",
            "num_workers": 2,
            // Define the label structure globally (for reproducability)
            "labels": [
                "BrainStem",
                "Chiasm",
                "Mandible",
                "OpticNerve_L",
                "OpticNerve_R",
                "Parotid_L",
                "Parotid_R",
                "Submandibular_L",
                "Submandibular_R"
            ],
            // Transformations to be applied to the labels only
            "label_transforms": [
                {
                    "name": "Transpose",
                    "dim_1": 0,
                    "dim_2": -1
                },
                {
                    "name": "CropAroundBrainStem",
                    "width": 256,
                    "height": 256,
                    "depth": 48
                }
            ],
            // Transformations to be applied to the feature only
            "sample_transforms": [
                {
                    "name": "Transpose",
                    "dim_1": 0,
                    "dim_2": -1
                },
                {
                    "name": "CropAroundBrainStem",
                    "width": 256,
                    "height": 256,
                    "depth": 48
                },
                {
                    "name": "StandardScaleTensor"
                }
            ]
        }
    }
}
```
## Special Mention

Special thanks for @prerakmody for providing the data (https://github.com/prerakmody/hansegmentation-uncertainty-qa/releases) and guidance!


## Acknowledgements

####[1] 
Chen, Z. et al. (2021). A Novel Hybrid Convolutional Neural Network for Accurate Organ Segmentation in 3D Head and Neck 
CT Images. In: , et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. MICCAI 2021. Lecture 
Notes in Computer Science(), vol 12901. Springer, Cham. https://doi.org/10.1007/978-3-030-87193-2_54

####[2] 
Raudaschl, P. F., Zaffino, P., Sharp, G. C., Spadea, M. F., Chen, A., Dawant, B. M., … & Jung, F. (2017).
Evaluation of segmentation methods on head and neck CT: Auto‐segmentation challenge 2015.
Medical Physics, 44(5), 2020-2036.

####[3] 
McGill University, Montreal, Canada - Special thanks to Martin Vallières of the Medical Physics Unit for providing it
