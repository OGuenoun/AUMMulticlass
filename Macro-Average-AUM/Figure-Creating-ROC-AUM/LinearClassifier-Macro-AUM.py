import torch 
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from CreatingAUM_Macro import *

df = pd.read_csv("C:/Users/nou-z/Downloads/mnist_train.csv/mnist_train.csv")

sampling_fractions = {
    1:0.0,
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

AUM_evolution=[]
# Training step
model.train()
probs = model(X)
auc_before=ROC_AUC(probs,y)
for epoch in range(10):
    loss = Proposed_AUM(probs, y)
    AUM_evolution.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    probs = model(X)
auc_after=ROC_AUC(probs,y)
print("before: ",auc_before)
print("after: ",auc_after)