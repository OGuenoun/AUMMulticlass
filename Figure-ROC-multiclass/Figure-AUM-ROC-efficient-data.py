import torch
import pandas as pd
import torch.nn.functional as F
#DATA
four_labels = torch.tensor([0,2,1,1])
four_pred = torch.tensor([[0.4, 0.3, 0.3],
                         [ 0.2, 0.1, 0.7],
                         [0.5,0.2,0.3],
                         [0.3,0.4,0.3]])
#Code for ROC and first step to AUM
def ROC_curve(pred_tensor, label_tensor, n_class):
    one_hot_labels = F.one_hot(label_tensor, num_classes=n_class)
    is_positive = one_hot_labels
    is_negative =1-one_hot_labels
    fn_diff = -is_positive
    fp_diff = is_negative
    fn_diff = fn_diff.flatten()
    fp_diff = fp_diff.flatten()
    thresh_tensor = -pred_tensor.flatten()
    fn_denom = is_positive.sum()
    fp_denom = is_negative.sum()
    sorted_indices = torch.argsort(thresh_tensor)
    sorted_fp_cum = fp_diff[sorted_indices].cumsum(0) / fp_denom
    sorted_fn_cum = -fn_diff[sorted_indices].flip(0).cumsum(0).flip(0) / fn_denom

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
        "FPR": FPR,
        "FNR": FNR,
        "TPR": 1 - FNR,
        "min(FPR,FNR)": torch.minimum(FPR, FNR),
        "min_constant": torch.cat([torch.tensor([-1], device=pred_tensor.device), uniq_thresh]),
        "max_constant": torch.cat([uniq_thresh, torch.tensor([1], device=pred_tensor.device)])
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

roc_efficient_df.to_csv("Figure-ROC-multiclass/ROC-efficient-points.csv")
