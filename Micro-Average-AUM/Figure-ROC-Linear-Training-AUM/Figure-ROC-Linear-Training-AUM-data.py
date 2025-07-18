import torch 
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def ROC_curve(pred_tensor, label_tensor):
    n_class=pred_tensor.size(1)
    one_hot_labels = F.one_hot(label_tensor, num_classes=n_class) 
    is_positive = one_hot_labels
    is_negative =1-one_hot_labels
    fn_diff = -is_positive.flatten()
    fp_diff = is_negative.flatten()
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
        "min_constant": torch.cat([torch.tensor([-1]), uniq_thresh]),
        "max_constant": torch.cat([uniq_thresh, torch.tensor([0])])
    }
#AUC
def ROC_AUC(pred_tensor, label_tensor):
    roc = ROC_curve(pred_tensor, label_tensor)
    FPR_diff = roc["FPR"][1:]-roc["FPR"][:-1]
    TPR_sum = roc["TPR"][1:]+roc["TPR"][:-1]
    return torch.sum(FPR_diff*TPR_sum/2.0)
#AUM 
def Proposed_AUM(pred_tensor, label_tensor):
    roc = ROC_curve(pred_tensor, label_tensor)
    min_FPR_FNR = roc["min(FPR,FNR)"][1:-1]
    constant_diff = roc["min_constant"][1:].diff()
    return torch.sum(min_FPR_FNR * constant_diff)

df = pd.read_csv("C:/Users/nou-z/Downloads/mnist_train.csv/mnist_train.csv")
X = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)/255 
y = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)

#Defining the linear model
class LinearClassifier_AUM(nn.Module):
    def __init__(self, input_dim, n_class):
        super(LinearClassifier_AUM, self).__init__()
        self.linear = nn.Linear(input_dim, n_class)
    
    def forward(self, x):
        logits = self.linear(x)
        probs = F.softmax(logits, dim=1) 
        return probs
    
model = LinearClassifier_AUM(input_dim=784, n_class=10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

df_list=[]
AUM_evolution=[]


# Training step
model.train()
probs = model(X)
df=ROC_curve(probs,y)
df['AUC']=ROC_AUC(probs,y)
df_list.append(df)
for epoch in range(600):
    loss = Proposed_AUM(probs, y)
    AUM_evolution.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    probs = model(X)
df=ROC_curve(probs,y)
df['AUC']=ROC_AUC(probs,y)
df_list.append(df)
list_1=[]
for df in df_list:
    df = pd.DataFrame({k: v.detach().numpy() if v.requires_grad else v.numpy() for k, v in df.items()})
    list_1.append(df)

list_1[0].to_csv("Micro-Average-AUM/Figure-ROC-Linear-Training-AUM/Initial_ROC_data.csv")
list_1[-1].to_csv("Micro-Average-AUM/Figure-ROC-Linear-Training-AUM/Final_ROC_data.csv")