"""ANN CLASSIFICATION"""

import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\arsha\Desktop\prime ai&ml\project\ann_classification\DateFruit_Dataset.csv")
print(df.head())
print(df.shape)

X = df.drop("Class" , axis = 1)
y = df["Class"]

print(df.Class.unique())

from sklearn.preprocessing import StandardScaler , LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size=0.3 , random_state=42
)

scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)

"""ANN"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , TensorDataset

X_train_tensor = torch.tensor(X_train_scaled , dtype = torch.float32)
y_train_tensor = torch.tensor(y_train , dtype = torch.long)

X_test_tensor = torch.tensor(X_test_scaled , dtype = torch.float32)
y_test_tensor = torch.tensor(y_test , dtype = torch.long)

train_dataset = TensorDataset(X_train_tensor , y_train_tensor)
test_dataset = TensorDataset(X_test_tensor , y_test_tensor)

train_loader = DataLoader(train_dataset  , batch_size = 32 , shuffle = True)
test_loader = DataLoader(test_dataset , batch_size = 32)

"""BUILD OUR MODEL"""

class ANN(nn.Module):
    def __init__(self):
        super(ANN , self).__init__()

        self.model = nn.Sequential(
            nn.Linear(X.shape[1] , 64),
            nn.ReLU(),
            nn.Linear(64 , 64),
            nn.ReLU(),
            nn.Linear(64 , 7)
        )

    def forward(self , x):
        return self.model(x)

model = ANN()

#loss and optim
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

"""Train the ANN"""

epochs = 100
for epoch in range(epochs):
    model.train()

    runing_loss = 0.0

    for xb , yb in train_loader:
        optimizer.zero_grad()

        outputs = model(xb)
        loss = criteria(outputs , yb)
        loss.backward()
        optimizer.step()

        runing_loss += loss.item()

    train_loss = runing_loss / len(train_loader)

    print(f"epoch = {epoch+1} / epochs , loss = {train_loss}")

"""EVALUATION"""
model.eval()

total = 0 
correct = 0

with torch.no_grad():
    for xb , yb in test_loader:
        outputs = model(xb) #return [0.1 , 0.2 , -1.5] this types of values
        _ , predicted = torch.max(outputs , 1)

        correct += (predicted == yb).sum().item()
        total += yb.size(0) #return actual sample in each batch
print("Total values : " , total)
print("correct values : " , correct)
print("accuracy : " , correct / total * 100)
