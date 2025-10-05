import torch 
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from CreatingAUM_Macro import *



def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()
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
df_test=pd.read_csv("C:/Users/nou-z/Downloads/mnist_test.csv/mnist_test.csv")
X_test=torch.tensor(df_test.iloc[:, 1:].values, dtype=torch.float32)/255 
y_test=torch.tensor(df_test.iloc[:, 0].values, dtype=torch.long)
lambdas=[0.001,0.01,0.1,1,10,100]
for lambd in lambdas:
    # Defining the imbalanced dataset
    sampling_fractions = {
    1:0.01,
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Training step for AUM
    acc=0
    probs = model(X)
    for epoch in range(1000):
        probs = model(X)
        loss = lambd*Proposed_AUM(probs, y)+loss_fn(probs,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc=get_accuracy(probs,y,y.size()[0])
    probs_test=model(X_test)
    AUM_AUC.append(ROC_AUC(probs_test,y_test).item())
    model = LinearClassifier_AUM(input_dim=784, num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    # Training step for CE
    
data_for_plotting=pd.DataFrame({
    'lambdas':lambdas,
    'AUC macro':AUM_AUC
})
data_for_plotting.to_csv("AUM_reg_CE.csv",index=False)