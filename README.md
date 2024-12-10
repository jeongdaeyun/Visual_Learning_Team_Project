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

### Evaluation data
Prepare vl checklist dataset as described in https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md  
Then move the vl dataset to `Visual-Representation-Learning/vl_datasets/`  
If you followed the instructions correctly, you should have the following folders inside vl_datasets: **'hake', 'swig', 'vg'**. 
그리고 ./evaluation을 참고하세요

First, navigate to the src directory:
```shell script
cd src
```

### Evaluation data
Prepare vl checklist dataset as described in https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md  
Then move the vl dataset to `Visual-Representation-Learning/vl_checklist_images_root_folder/`  
If you followed the instructions correctly, you should have the following folders inside vl_datasets: **'hake', 'swig', 'vg'**. 

prepare aro dataset as described in https://github.com/mertyg/vision-language-models-are-bows
Then move the aro dataset to `Visual-Representation-Learning/aro/` 

### Run the training script

The model will be saved in `Visual-Representation-Learning/src/checkpoints`

To train a network with quality captions and:
* training ours model:
```shell script
python3 training/main.py --epochs 25 --name exp_name --lora 4 --batch-size 10 --vl_negs --neg_type rand_both --auto_neg_types NOUN ADP ADJ VERB --mil_batch 10 --pretrained openai
```

## Evaluation
### Run the evaluation script
####you can download our checkpoints of Visual-Representation-Learning_SAM and Visual-Representation-Learning_LLM from here: https://drive.google.com/drive/folders/1bUpGOa2_tsDlg2B2peVayISbI6epCT1x?usp=sharing

All vl_checklist jsons will be saved in `Visual-Representation-Learning/eval_jsons/clip/exp_name/` and the result will be printed. 
To prepare the vl checklist evaluate results for the experiment **exp_name** run the following command:
```shell script
mkdir vl_checklist_accuracy_jsons_folder
python3 training/main.py  --lora 4 --pretrained openai --eval_vl_cklist --eval_only --resume /path/to/checkpoint --vl_checklist_images_root_folder Visual-Representation-Learning/vl_checklist_images_root_folder/
```

To print the aro evaluated results for the experiment **exp_name** run the following command:
```shell script
python3 aro_clip_lora_eval.py  --lora 4 --resume /path/to/checkpoint
```
