import torch 
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from CreatingAUM_Macro import *

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
acc=0
model.train()
probs = model(X)
auc_before=ROC_AUC(probs,y,10)
df['AUC']=ROC_AUC(probs,y,10)
df_list.append(df)
for epoch in range(600):
    loss = Proposed_AUM(probs, y,10)
    AUM_evolution.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    probs = model(X)
auc_after=ROC_AUC(probs,y,10)
print("before: ",auc_before)
print("after: ",auc_after)