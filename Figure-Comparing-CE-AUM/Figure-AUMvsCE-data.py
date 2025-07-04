import torch 
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
#Defining useful functions

def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

def ROC_curve(pred_tensor, label_tensor,n_class):
    is_positive = label_tensor == 0
    is_negative = label_tensor != 0
    fn_diff = torch.where(is_positive, -1, 0)
    fp_diff = torch.where(is_positive, 0, 1)
    fp_denom = torch.sum(is_negative) 
    fn_denom = torch.sum(is_positive)
    thresh_tensor = (0.5-pred_tensor[:,0]).flatten()
    for i in range(1,n_class):
        is_positive = label_tensor == i
        is_negative = label_tensor != i
        fn_diff = torch.cat([fn_diff,torch.where(is_positive, -1, 0)])
        fp_diff = torch.cat([fp_diff,torch.where(is_positive, 0, 1)])
        thresh_tensor = torch.cat([thresh_tensor,(0.5-pred_tensor[:,i]).flatten()])
        fp_denom += torch.sum(is_negative)
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

def ROC_AUC(pred_tensor, label_tensor,n_class):
    roc = ROC_curve(pred_tensor, label_tensor,n_class)
    FPR_diff = roc["FPR"][1:]-roc["FPR"][:-1]
    TPR_sum = roc["TPR"][1:]+roc["TPR"][:-1]
    return torch.sum(FPR_diff*TPR_sum/2.0)

def Proposed_AUM(pred_tensor, label_tensor,n_class):

    roc = ROC_curve(pred_tensor, label_tensor,n_class)
    min_FPR_FNR = roc["min(FPR,FNR)"][1:-1]
    constant_diff = roc["min_constant"][1:].diff()
    return torch.sum(min_FPR_FNR * constant_diff)

#Defining the model

class LinearClassifier_AUM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier_AUM, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        logits = self.linear(x)
        probs = F.softmax(logits, dim=1) 
        return probs
#Initializing a list to stock different AUCs from different datasets
AUM_AUC=[]
CE_AUC=[]
loss_fn=nn.CrossEntropyLoss()
df = pd.read_csv("C:/Users/nou-z/Downloads/mnist_train.csv/mnist_train.csv")
for i in range(10):
    # Defining the imbalanced dataset
    sampling_fractions = {
    i:0.01,
    }
    df_imbalanced = pd.concat([
        df[df.label == clas].sample(
            frac=sampling_fractions.get(clas, 1.0),
            random_state=42
        )
        for clas in df['label'].unique()
    ])
    X = torch.tensor(df_imbalanced.iloc[:, 1:].values, dtype=torch.float32)/255 
    y = torch.tensor(df_imbalanced.iloc[:, 0].values, dtype=torch.long)

    #Initializing the model
        
    model = LinearClassifier_AUM(input_dim=784, num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    # Training step for AUM
    acc=0
    probs = model(X)
    for epoch in range(500):
        probs = model(X)
        loss = Proposed_AUM(probs, y,10)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc=get_accuracy(probs,y,y.size()[0])
    AUM_AUC.append(ROC_AUC(probs,y,10))
    model = LinearClassifier_AUM(input_dim=784, num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    # Training step for CE

    for epoch in range(500):
        probs = model(X)
        loss = loss_fn(probs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc=get_accuracy(probs,y,y.size()[0])
    CE_AUC.append(ROC_AUC(probs,y,10))
    print(f"Finished {i} try")
data_for_plotting=pd.DataFrame({
    'AUM':AUM_AUC,
    'Cross Entropy':CE_AUC
})
data_for_plotting.to_csv("AUMvsCE.csv")