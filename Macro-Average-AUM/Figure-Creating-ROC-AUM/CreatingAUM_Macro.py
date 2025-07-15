import torch 
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
#DATA
four_labels = torch.tensor([0,2,1,1])
four_pred = torch.tensor([[0.4, 0.3, 0.3],
                         [ 0.2, 0.1, 0.7],
                         [0.5,0.2,0.3],
                         [0.3,0.4,0.3]])


#Defining useful functions

def ROC_curve(pred_tensor, label_tensor):
    n_class=pred_tensor.size(1)
    one_hot_labels = F.one_hot(label_tensor, num_classes=n_class)
    is_positive = one_hot_labels
    is_negative =1-one_hot_labels
    fn_diff = -is_positive
    fp_diff = is_negative
    thresh_tensor = -pred_tensor
    fn_denom = is_positive.sum(dim=0)
    fp_denom = is_negative.sum(dim=0)
    sorted_indices = torch.argsort(thresh_tensor,dim=0)
    sorted_fp_cum = torch.div(torch.gather(fp_diff, dim=0, index=sorted_indices).cumsum(0), fp_denom)
    sorted_fn_cum = -torch.div(torch.gather(fn_diff, dim=0, index=sorted_indices).flip(0).cumsum(0).flip(0) , fn_denom)
    sorted_thresh = torch.gather(thresh_tensor, dim=0, index=sorted_indices)
    #Problem starts here 
    zeros_vec=torch.zeros(1,n_class)
    FPR = torch.cat([zeros_vec, sorted_fp_cum])
    FNR = torch.cat([sorted_fn_cum, zeros_vec])
    return {
        "FPR_all_classes": FPR,
        "FNR_all_classes": FNR,
        "TPR_all_classes": 1 - FNR,
        "min(FPR,FNR)": torch.minimum(FPR, FNR),
        "min_constant": torch.cat([-torch.ones(1,n_class), sorted_thresh]),
        "max_constant": torch.cat([sorted_thresh, zeros_vec])
    }

def ROC_AUC(pred_tensor, label_tensor):
    roc = ROC_curve(pred_tensor, label_tensor)
    FPR_diff = roc["FPR_all_classes"][1:,:]-roc["FPR_all_classes"][:-1,]
    TPR_sum = roc["TPR_all_classes"][1:,:]+roc["TPR_all_classes"][:-1,:]
    auc= torch.sum(FPR_diff*TPR_sum/2.0,dim=0)
    return torch.mean(auc)

def Proposed_AUM(pred_tensor, label_tensor):

    roc = ROC_curve(pred_tensor, label_tensor)
    min_FPR_FNR = roc["min(FPR,FNR)"][1:-1,:]
    constant_diff = roc["min_constant"][1:,:].diff(dim=0)
    aum= torch.sum(min_FPR_FNR * constant_diff,dim=0)
    return torch.mean(aum)