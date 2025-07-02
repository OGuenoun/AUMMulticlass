# AUM Multi-class

*(The code in this repo was inspired by [this blog](https://tdhock.github.io/blog/2024/torch-roc-aum/))*

## 1. Introduction

In this repository, I aim to tackle the problem of optimizing the ROC AUC in the context of multiclass classification. The approach used is One-vs-All (OvA), where the multiclass problem is reduced to multiple binary classification problems.

## 2. ROC (Receiver Operating Characteristic)

I began this repo by implementing the code necessary to generate ROC curves, using micro-averaging to aggregate the True Positive Rates (TPRs) and False Positive Rates (FPRs) from the OvA classifiers. I visualized it using a simple dataset with 4 labels, 3 classes and 1 feature 
<p align="center">
  <img src="ROC_multiclass_micro_plot.png" alt="Description" width="400"/>
</p>
