import torch
import pandas as pd
#DATA
four_labels = torch.tensor([0,2,1,1])
four_pred = torch.tensor([[0.4, 0.3, 0.3],
                         [ 0.2, 0.1, 0.7],
                         [0.5,0.2,0.3],
                         [0.3,0.4,0.3]])
#Code for ROC and first step to AUM
def ROC_curve(pred_tensor, label_tensor,n_class):
    is_positive = label_tensor == 0
    is_negative = label_tensor != 0
    fn_diff = torch.where(is_positive, -1, 0)
    fp_diff = torch.where(is_positive, 0, 1)
    fp_denom = torch.sum(is_negative) #or 1 for AUM based on count instead of rate
    fn_denom = torch.sum(is_positive)
    thresh_tensor = (0.5-pred_tensor[:,0]).flatten()
    for i in range(1,n_class):
        is_positive = label_tensor == i
        is_negative = label_tensor != i
        fn_diff = torch.cat([fn_diff,torch.where(is_positive, -1, 0)])
        fp_diff = torch.cat([fp_diff,torch.where(is_positive, 0, 1)])
        thresh_tensor = torch.cat([thresh_tensor,(0.5-pred_tensor[:,i]).flatten()])
        fp_denom += torch.sum(is_negative) #or 1 for AUM based on count instead of rate
        fn_denom += torch.sum(is_positive)
    sorted_indices = torch.argsort(thresh_tensor)
    sorted_fp_cum = fp_diff[
        sorted_indices].cumsum(axis=0)/fp_denom
    sorted_fn_cum = -fn_diff[
        sorted_indices].flip(0).cumsum(axis=0).flip(0)/fn_denom
    sorted_thresh = thresh_tensor[sorted_indices]
    sorted_is_diff = sorted_thresh.diff() != 0
    sorted_fp_end = torch.cat([sorted_is_diff, torch.tensor([True])])
    sorted_fn_end = torch.cat([torch.tensor([True]), sorted_is_diff])
    uniq_thresh = sorted_thresh[sorted_fp_end]
    uniq_fp_after = sorted_fp_cum[sorted_fp_end]
    uniq_fn_before = sorted_fn_cum[sorted_fn_end]
    FPR = torch.cat([torch.tensor([0.0]), uniq_fp_after])
    FNR = torch.cat([uniq_fn_before, torch.tensor([0.0])])
    return {
        "FPR":FPR,
        "FNR":FNR,
        "TPR":1 - FNR,
        "min(FPR,FNR)":torch.minimum(FPR, FNR),
        "min_constant":torch.cat([torch.tensor([-0.5]), uniq_thresh]),
        "max_constant":torch.cat([uniq_thresh,torch.tensor([0.5])])
    }
roc_efficient_df = pd.DataFrame(ROC_curve(four_pred, four_labels,3))
#AUC
def ROC_AUC(pred_tensor, label_tensor,n_class):
    roc = ROC_curve(pred_tensor, label_tensor,n_class)
    FPR_diff = roc["FPR"][1:]-roc["FPR"][:-1]
    TPR_sum = roc["TPR"][1:]+roc["TPR"][:-1]
    return torch.sum(FPR_diff*TPR_sum/2.0)
#AUM 
def Proposed_AUM(pred_tensor, label_tensor,n_class):

    roc = ROC_curve(pred_tensor, label_tensor,n_class)
    min_FPR_FNR = roc["min(FPR,FNR)"][1:-1]
    constant_diff = roc["min_constant"][1:].diff()
    return torch.sum(min_FPR_FNR * constant_diff)


