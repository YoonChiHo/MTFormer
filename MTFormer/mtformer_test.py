# Python built-in
import torch
import os 
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
# Import code
from datasets import DefectDatasetV3
from networks.networks import MTL2d_DeepSVDD
from os.path import dirname, basename
def svdd_test(args, logging, target_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_datasets = {'test': DefectDatasetV3(data_dir=args.data_dir, image_size=args.input_size, transform_mode='test', normal_classes = args.normal_classes, abnormal_classes = args.abnormal_classes, in_channels = args.in_channels)}
        
    dataloaders = {x:  DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() and device != 'cpu' else False)
                        for x in ['test']}

    logging.info(f"Num of test: {len(image_datasets['test'])}")

    cls_info = {"names": image_datasets['test'].classes, "n_idx":image_datasets['test'].normal_idx, "an_idx":image_datasets['test'].abnormal_idx}
    
    num_class = len(image_datasets['test'].normal_idx)

    # Model Setting
    if os.path.isfile(os.path.join(args.result_dir, args.task_name, f'model_params_svdd_{target_model}.pt')):    
        best_model_params_path = os.path.join(args.result_dir, args.task_name, f'model_params_svdd_{target_model}.pt')
    else:
        logging.info(f'Error: no saved {target_model} model parameter')
        return
    state_dict = torch.load(best_model_params_path)
    center = torch.Tensor(state_dict['center']).to(device)
    radius = state_dict['radius']
    
    num_class = state_dict['model_dict']['cls.linear.weight'].shape[0]
    svdd = MTL2d_DeepSVDD(num_class, args.att_h, args.att_l, args.num_layer, args.input_size, in_channels = args.in_channels).to(device) #for multitasking

    # Load checkpoint
    svdd.load_state_dict(state_dict['model_dict'], strict=False) 

    def test_model(model, target_model, C, R):
        x_ = []
        z_ = []
        y_ = []
        score_ = []
        paths_=[]
        p_cls_=[]
        dist_ = [] ###
        # Iterate over data.
        for inputs, labels, name in dataloaders['test']:
            inputs = inputs.to(device)

            outputs = model(inputs) #output-feature/classification
            z = outputs[0]          # output-feature
            dist = torch.sum((z - C)**2, 1) # distance from z to C
            score = dist
            # Convert to a value between 0 and 1 with a maximum value of 2*R^2
            score[score > 2 * (R ** 2)] = torch.Tensor([2 * (R ** 2)]).to(device)
            score = score / (2 * (R ** 2))
            
            score_.append(score.cpu().detach()) # score list
            dist_.append(dist.cpu().detach()) ###
            paths_.extend(name)                 # path list

            x_.append(inputs.detach())  #input list
            z_.append(z.detach())       #feature list
            y_.extend(labels)           #label list
            #cls score
            p_cls_.append(outputs[1].detach())

        # Parameter setting for visualization
        dist_.append(torch.unsqueeze(R.cpu().detach() ** 2, 0))
        x_, z_, score_, dist_ = torch.cat(x_), torch.cat(z_), torch.cat(score_), torch.cat(dist_) ###
        zc_ = torch.cat((torch.squeeze(z_), torch.unsqueeze(C,0)))
        yc_ = y_+[-1]

        #Evaluate Code
        pred_ = np.where(score_>0.5,1,0)  # relative to R, 1 (abnormal) if dist is larger, 0 (normal) if smaller
        yf_ = np.where(np.isin(y_, cls_info['an_idx']),1,0)  # set label to 1 for abnormal classes and label to 0 for normal

        # Each Class Results
        # normal case
        for n_idx in cls_info["n_idx"]:     # Normal Class
            name = cls_info["names"][n_idx]
            pred_target = []
            score_target = []
            y_target = []
            path_target = []
            for iy in range(len(y_)):
                if y_[iy] == n_idx:         # Target normal class
                    pred_target.append(pred_[iy])
                    score_target.append(score_[iy])
                    y_target.append(0)             
                    path_target.append(paths_[iy])
            out  = confusion_matrix(np.array(y_target), pred_target).ravel()

            if len(out) == 1:   # Exception
                TN, FP, FN, TP = int(out), 0, 0, 0
                FPR = 0
                FNR = 0
            else:
                TN, FP, FN, TP = out
                FPR = FP/(TN+FP)
                FNR = FN/(TP+FN)
            logging.info(f'[Normal {name}] TP: {TP} FP: {FP} TN: {TN} FN: {FN}, FPR:{FPR:.5f}, FNR:{FNR:.5f}')

        for an_idx in cls_info["an_idx"]:   # Abnormal Class
            name = cls_info["names"][an_idx]
            pred_target = []
            score_target = []
            y_target = []
            path_target = []
            for iy in range(len(y_)):
                if y_[iy] == an_idx:
                    pred_target.append(pred_[iy])
                    score_target.append(score_[iy])
                    y_target.append(1)              # AbNormal Class이므로 1으로 설정
                    path_target.append(paths_[iy])
            out  = confusion_matrix(np.array(y_target), pred_target).ravel()
            
            if len(out) == 1:  # exception
                TN, FP, FN, TP = 0, 0, 0, int(out)
                FPR = 0
                FNR = 0
            else:
                TN, FP, FN, TP = out
                FPR = FP/(TN+FP)
                FNR = FN/(TP+FN)
            logging.info(f'[Anormal {name}] TP: {TP} FP: {FP} TN: {TN} FN: {FN}, FPR:{FPR:.5f}, FNR:{FNR:.5f}')

        logging.info(score_)
        logging.info(pred_)
        logging.info(yf_)
        TN, FP, FN, TP  = confusion_matrix(np.array(yf_), pred_).ravel()
        logging.info(f'[{target_model}] ROC AUC score: {roc_auc_score(yf_, score_)*100:.2f}, TP: {TP} FP: {FP} TN: {TN} FN: {FN}, FPR:{FP/(TN+FP):.5f}, FNR:{FN/(TP+FN):.5f}')
        
        return target_model, roc_auc_score(yf_, score_)*100, TP, FP, TN, FN

    target_model, roc, TP, FP, TN, FN = test_model(svdd, target_model, center, radius)
    return target_model, roc, TP, FP, TN, FN
