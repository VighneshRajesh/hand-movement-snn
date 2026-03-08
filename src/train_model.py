import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
from lif_snn_model import GestureSNN

with open("dataset.json") as f:
    data = json.load(f)

X = data["X"]
y = data["y"]

combined = list(zip(X, y))
random.shuffle(combined)

X, y = zip(*combined)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1,1)

print("Dataset shape:", X.shape)

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

model = GestureSNN()

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(400):

    optimizer.zero_grad()

    output = model(X_train)

    loss = criterion(output, y_train)

    loss.backward()

    optimizer.step()

    if epoch % 20 == 0:
        print("Epoch", epoch, "Loss", loss.item())


with torch.no_grad():

    preds = model(X_test)

    predicted = (preds > 0.5).float()

    accuracy = (predicted == y_test).sum() / len(y_test)

print("Test Accuracy:", accuracy.item())

torch.save(model.state_dict(),"gesture_snn.pth")

print("Model saved")