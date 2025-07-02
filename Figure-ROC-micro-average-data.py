import pandas as pd
import torch
#### First part: Drawing the ROC curve:
# I took a small dataset as M.Hocking did in his blog
four_labels = torch.tensor([0,2,1,1])
four_pred = torch.tensor([[0.4, 0.3, 0.3],
                         [ 0.2, 0.1, 0.7],
                         [0.5,0.2,0.3],
                         [0.3,0.4,0.3]])
## Defining functions to do the OvA( One versus all) comparison:
def probabilty_of_pred(pred_score,class_n):
    return pred_score[:,class_n]


def OvA_classi(pred_score,class_n):
    return torch.where(pred_score[:,class_n] < 0.5, -1, 1)

def pred_is_correct_OvA(pred_score, label_vec,class_n):
    pred_class = OvA_classi(pred_score,class_n)
    return (((pred_class == 1) & (label_vec==class_n))|((pred_class == -1) & (label_vec!=class_n)))

def zero_one_loss(pred_score, label_vec,class_n):
    return torch.where(pred_is_correct_OvA(pred_score, label_vec,class_n), 0, 1)
def label_OvA(label_vec,class_n):
    return torch.where(label_vec ==class_n, 1, -1)

zero_one_df_list =[] 
for i in range(3):
    new_labels=label_OvA(four_labels,i)
    zero_one_df=pd.DataFrame({
    "label":four_labels,
    "label_OvA": new_labels,
    "pred":OvA_classi(four_pred,i),
    "zero_one_loss":zero_one_loss(four_pred, four_labels,i)
    })
    print(zero_one_df)
    zero_one_df_list.append(zero_one_df)

## Functions to compute TPR and FPR both micro and for each class when doing the OvA
def get_TPR_per_class(df,true_class):
    return ((df.pred==1) & (df.label==true_class)).sum()/(df.label==true_class).sum()

def get_TPR_micro(df_list):
    c1=0
    c2=0
    for i in range(len(df_list)):
        c1+=((df_list[i].pred==1) & (df_list[i].label==i)).sum()
        c2+=(df_list[i].label==i).sum()
    return c1/c2



def get_FPR_per_class(df,true_class):
    return ((df.pred==1) & (df.label!=true_class)).sum()/(df.label!=true_class).sum()

def get_FPR_micro(df_list):
    c1=0
    c2=0
    for i in range(len(df_list)):
        c1+=((df_list[i].pred==1) & (df_list[i].label!=i)).sum()
        c2+=(df_list[i].label!=i).sum()
    return c1/c2

## Defining functions useful for varying the threshold

def error_one_constant(constant,class_n):
    pred_const = probabilty_of_pred(four_pred,class_n)+constant
    pred_const_vect= torch.zeros_like(four_pred)
    pred_const_vect[:,class_n]=pred_const
    return pd.DataFrame({
        "label":four_labels,
        "probability":probabilty_of_pred(four_pred,class_n),
        "probability_plus_constant":pred_const,
        "pred":OvA_classi((pred_const_vect),class_n),
        "zero_one_loss":zero_one_loss(pred_const_vect, four_labels,class_n)
    })

all_const_vec=[]
for i in range(3):
    constant_vec = list(0.5-probabilty_of_pred(four_pred,i))
    all_const_vec+=constant_vec
all_const_vec+=[-0.5]
all_const_vec.sort()

def one_roc_point(constant):
    one_df_list = []
    for i in range(3):
        one_df=error_one_constant(constant,i)
        one_df_list.append(one_df)
    return pd.DataFrame({
        "constant":[float(constant)],
        "TPR":get_TPR_micro(one_df_list),
        "FPR":get_FPR_micro(one_df_list),
    })
roc_inefficient_df = pd.concat([
    one_roc_point(constant) for constant in all_const_vec
])
roc_inefficient_df.to_csv('ROC-multiclass-points.csv', index=False)