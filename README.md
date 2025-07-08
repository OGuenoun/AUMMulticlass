# [AUM](https://www.jmlr.org/papers/v24/21-0751.html) Multi-class



## 1. Introduction

In this repository, I aim to tackle the problem of optimizing the ROC AUC in the context of multiclass classification. The approach used is One-vs-All (OvA), where the multiclass problem is reduced to multiple binary classification problems.

## 2. ROC (Receiver Operating Characteristic) *(The code in this part was inspired by [this blog](https://tdhock.github.io/blog/2024/torch-roc-aum/))*

I began this repo by implementing the code necessary to generate ROC curves, using micro-averaging to aggregate the True Positive Rates (TPRs) and False Positive Rates (FPRs) from the OvA classifiers. I visualized it using a simple dataset with 4 labels, 3 classes and 1 feature 
<p align="center">
  <img src="Micro-Average-AUM/Figure-ROC-multiclass/ROC_multiclass_micro_plot.png" alt="Description" width="400"/>
</p>

## 3. A first model : linear classifier
I trained a linear classifier using MNIST data using AUM as a loss function ( micro-averaging the OvA AUM) , I then drew the ROC curve before and after training.The ROC AUC seems to increase when the AUM decreases : The initial AUC was 0.53 , the optimized AUC is 0.94
<p align="center">
  <img src="Micro-Average-AUM/Figure-ROC-Linear-Training-AUM/ROC_Linear_Training_AUM.png" alt="Description" width="400"/>
</p>
3.1 Comparing two loss functions : Cross-entropy and AUM
<p align="center">
  <img src="Figure-Comparing-CE-AUM\AUMvsCE.png" alt="Description" width="400"/>
</p>
To get this figure , I trained a linear classifier on the MNIST dataset keeping at each training run only 1% of each class , this means I trained the linear classifier 10 times . In this figure , Cross-entropy (unweighted) seems to give better AUC scores than the AUM even when trained on imbalanced datasets which was surprising to me because the unweighted cross-entropy doesn't do well when trained on imbalanced datasets