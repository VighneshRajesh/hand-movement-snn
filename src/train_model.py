import torch
import torch.nn as nn
import torch.optim as optim
import json
import random

with open("dataset.json") as f:
    data = json.load(f)

X = data["X"]
y = data["y"]

# Shuffle dataset
combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1,1)

print("Dataset size:", X.shape)

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


class GestureNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(8,16)
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,1)

    def forward(self,x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


model = GestureNet()

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(400):

    optimizer.zero_grad()

    output = model(X_train)

    loss = criterion(output, y_train)

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print("Epoch:", epoch, "Loss:", loss.item())


with torch.no_grad():

    preds = model(X_test)

    predicted = (preds > 0.5).float()

    accuracy = (predicted == y_test).sum() / len(y_test)

print("Test Accuracy:", accuracy.item())

torch.save(model.state_dict(),"gesture_model.pth")

print("Model saved")