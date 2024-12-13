# Visual-Representation-Learning Team Project
## Our Method
![image](https://github.com/user-attachments/assets/bfb882ee-b454-4c83-a938-6fedbca41cdd)

_______________________________
## Environment
```shell script
conda deactivate # deactivate any active environments
conda create -n region python=3.8.13 # install the conda environment with conda dependencies
conda activate region # activate the environment
conda install -c conda-forge libjpeg-turbo
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3.1 -c pytorch
pip install -r requirements.txt
```

## Data Preparations and creation 
### Training Data
#### Download Ours Data
Download Ours Dataset training and validation splits from https://drive.google.com/drive/folders/1bUpGOa2_tsDlg2B2peVayISbI6epCT1x?usp=sharing
After data preparation, place the data in `Visual-Representation-Learning/train` and `Visual-Representation-Learning/val`  

Download and place in `Visual-Representation-Learning/` our_train_augment_dataset.csv and val_with_cap.csv from `Visual-Representation-Learning//VG_data`

First, navigate to the src directory:
```shell script
cd src
```

### Evaluation data
prepare ours dataset (https://drive.google.com/drive/folders/1bUpGOa2_tsDlg2B2peVayISbI6epCT1x?usp=sharing)

Prepare vl checklist dataset as described in https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md  
vl_datasets: **'hake', 'swig', 'vg'**. 

prepare aro dataset as described in https://github.com/mertyg/vision-language-models-are-bows

1. Place the datasets ours, aro, and vl_checklist in the folder.
2. Set the path to the datasets in the Python scripts under the evaluation folder.

### Run the training script

The model will be saved in `Visual-Representation-Learning/src/checkpoints`

To train a network with quality captions and:
* training ours model:
```shell script
python3 training/main.py --epochs 25 --name exp_name --lora 4 --batch-size 10 --vl_negs --neg_type rand_both --auto_neg_types NOUN ADP ADJ VERB --mil_batch 10 --pretrained openai
```
## result
![image](https://github.com/user-attachments/assets/13f87f3a-0e19-4127-a04f-dc67418b9413)

## Evaluation
### Run the evaluation script
#### Ours Region Dataset
1. Please download the dataset from the following link:
[Dataset Download](https://drive.google.com/drive/folders/1bUpGOa2_tsDlg2B2peVayISbI6epCT1x?usp=sharing)
2. Place check region_level_dataset the JSON files, images, and model weights in the evaluation
* evaluation Ours Region:
```shell script
python3 evaluation/region_level_dataset/region_level.py
```
#### Global ARO Dataset
1. If you need to download the ARO Dataset programmatically, you can use a script to automate the process
2. Place check ARO_dataset the JSON files, images, and model weights in the evaluation
* evaluation ARO:
```shell script
python3 evaluation/ARO_dataset/VG_attribute.py
python3 evaluation/ARO_dataset/VG_relation.py
```
#### Global VL-CheckList Dataset
1. If you need to download the VL-CheckList Dataset programmatically, you can use a script to automate the process.
2. Place check VL_checklist_dataset the JSON files, images, and model weights in the evaluation
* evaluation VL_CheckList:
```shell script
python3 evaluation/VL_checklist/VL_object.py
python3 evaluation/VL_checklist/VL_attribute.py
python3 evaluation/VL_checklist/VL_relation.py
```
