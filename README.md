# MTFormer: Multi-Task Hybrid Transformer and Deep Support Vector Data Description to Detect Novel anomalies during Semiconductor Manufacturing

This repository provides the code for **MT-Former**, a model that uses a Multi-Task Hybrid Transformer integrated with Deep Support Vector Data Description (Deep SVDD) for detecting novel anomalies. This implementation includes preprocessing scripts, main model code, and a sample run script to help reproduce the experiments.

## Contents
- `public_preprocess.py`: Script to preprocess data.
- `MTFormer/`: Main code folder containing the MT-Former model implementation.
- Sample execution scripts for testing and inference.

## Trained Model Checkpoint
To replicate the experiments, download the pre-trained model checkpoint from [here](https://1drv.ms/u/c/de011cb09ae2716d/EeussqGxjNVAjqhqHaPiJLUBgPJClEG2VUipONG1GnsaUw?e=vqgNdp). Extract the contents and place the `logs` folder at the same level as the `MTFormer` folder.

## Getting Started

### Step-by-Step Guide for Reproducing Experiments

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YoonChiHo/MTFormer /{target_folder}
2. **Download Model Parameters**
Download the model parameters from the [Link]([https://medmnist.com/v1](https://1drv.ms/u/c/de011cb09ae2716d/EeussqGxjNVAjqhqHaPiJLUBgPJClEG2VUipONG1GnsaUw?e=vqgNdp))  
Extract them, and place the logs folder in the same directory as MTFormer.  

4. **Set Up Docker Environment**
Install Docker from docker.com if not already installed.
Run the following commands to create and access a Docker container: 
- Create Docker container: 
   ```bash
   docker run -dit --name run_mtformer --gpus all --shm-size 256g -v /{target_folder}:/workspace pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
- Attach to Docker container:
   ```bash
   docker attach run_mtformer'
  
4. **Data Preprocessing**
Use public_preprocess.py to preprocess the HAM10000 dataset ([DermaMNIST](https://medmnist.com/v1)), which can be downloaded from MedMNIST.
The script will convert the dataset into a PNG image format and split it into train and test sets.  
   ```bash
   python public_preprocess.py

6. **Install Required Packages**
Navigate to the MTFormer folder and install the necessary packages for inference:  
   ```bash
   pip install -r MTFormer/requirements.txt

7. **Run Sample Code**
Use the following sample code for inference, where Class 0 is set as Abnormal and the remaining classes are set as Normal:  
   ```bash
   python MTFormer/main.py \
   -t mtformer --result_dir "logs/derma_c3_0" \
   --att_h 3 --att_l 0 1 2 --num_layer 3 \
   --input_size 32 -e_svdd 100 --in_channels 3 \
   --data_dir "data/derma" -n 1 2 3 4 5 6 -an 0

References
1. Yang, Jiancheng, et al. "Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification." Scientific Data 10.1 (2023): 41.
2. Philipp Tschandl, Cliff Rosendahl, and Harald Kittler, "The HAM10000 dataset, a large collection of multisource dermatoscopic images of common pigmented skin lesions," Scientific Data, vol. 5, pp. 180161, 2018.
3. Noel Codella, Veronica Rotemberg, et al.: “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”, 2018; arXiv:1902.03368.
4. [MedMNIST](https://medmnist.com/v1)

