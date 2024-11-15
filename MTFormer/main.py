import os
import argparse
import torch
import numpy as np
import logging
from mtformer_test import svdd_test
import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser()
# Basic Setting
parser.add_argument('-t', '--task_name',                    default='test')
parser.add_argument('--gpus', type = str,                   default='0')                        # Set the gpu number (CPU: '', GPU0: '0')
parser.add_argument('--randseed',type = int,                default=50)          
# Dataset Setting
parser.add_argument('--data_dir',                           default=f'./data/derma')     # Dataset folder
parser.add_argument('--result_dir',                         default=f'logs')                    # Save folder
parser.add_argument('-n', '--normal_classes', nargs='+',    default=['1','2','3','4','5','6'])  # Normal Class Names
parser.add_argument('-an', '--abnormal_classes', nargs='+', default=['0'])                      # Abnormal Class Names
parser.add_argument('--in_channels',type = int,             default=3)                          # gray to 1 / rgb to 3
parser.add_argument('--input_size', type = int,             default=32)                         # Target input size (resize to input_size)
# TEST Setting
parser.add_argument('-e_svdd','--num_epochs_svdd',type = int,default=100)                       # Nmb of training epochs for SVDD
parser.add_argument('--num_workers',type = int,             default=10)                         # Nmb of workers
# Model Setting
parser.add_argument('--num_layer',type = int,               default=3)                          # Depth of model
parser.add_argument('--att_h',type = int,                   default=3)                          # Classification loss weight for CAE (If 0, it is not MTL)
parser.add_argument('--att_l', type = int, nargs='+',       default=[])                         # [] : no att

args = parser.parse_args()

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
# Set deterministic
random_seed = args.randseed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(random_seed)

print(f'Normal Classes : {len(args.normal_classes)}')
print(f'ABNormal Classes : {len(args.abnormal_classes)}')

# Make Dirs
os.makedirs(os.path.join(args.result_dir, args.task_name), exist_ok= True)

# Log Setting
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

log_file_path = f'{args.result_dir}/{args.task_name}/logs_{datetime.today().strftime("%Y%m%d_%H%M")}.txt'
with open(log_file_path, 'w'):
    pass
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logging.info("\n\nStep3 SVDD_test started")

results = {'target_model':[], 'FP/(FP+TN)':[], 'FN/(TP+FN)':[], 'roc':[], 'TP':[], 'FP':[], 'TN':[], 'FN':[]}

logging.info(f"\nInference e{args.num_epochs_svdd-1:03d} Model")
svdd_results = svdd_test(args, logging, f'e{args.num_epochs_svdd-1:03d}')  # step 3
results['target_model'].append(svdd_results[0])
results['FP/(FP+TN)'].append(svdd_results[3]/(svdd_results[4]+svdd_results[3]))
results['FN/(TP+FN)'].append(svdd_results[5]/(svdd_results[2]+svdd_results[5]))
results['roc'].append(svdd_results[1])
results['TN'].append(svdd_results[4])
results['FP'].append(svdd_results[3])
results['TP'].append(svdd_results[2])
results['FN'].append(svdd_results[5])
    
# Save all results in csv format
df = pd.DataFrame(results)
df.to_csv(os.path.join(args.result_dir, args.task_name, 'svdd_scores.csv'), index = False)
