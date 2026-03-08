import torch
import torch.nn as nn
import torch.optim as optim
import pickle


# -------------------------
# Load dataset
# -------------------------

with open("gesture_dataset.pkl","rb") as f:
    X,y = pickle.load(f)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


# -------------------------
# LIF neuron
# -------------------------

class LIFNeuron(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.threshold = 1.0
        self.decay = 0.9
        self.size = size

    def forward(self, x):

        mem = torch.zeros_like(x)
        spikes = torch.zeros_like(x)

        mem = self.decay * mem + x
        spikes = (mem > self.threshold).float()

        mem = mem * (1 - spikes)

        return spikes


# -------------------------
# SNN model
# -------------------------

class SNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(8,16)
        self.lif1 = LIFNeuron(16)

        self.fc2 = nn.Linear(16,16)
        self.lif2 = LIFNeuron(16)

        self.fc3 = nn.Linear(16,1)

    def forward(self,x):

        x = self.fc1(x)
        x = self.lif1(x)

        x = self.fc2(x)
        x = self.lif2(x)

        x = torch.sigmoid(self.fc3(x))

        return x


# -------------------------
# Training setup
# -------------------------

model = SNN()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# -------------------------
# Training loop
# -------------------------

epochs = 100

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(outputs,y)

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch:",epoch,"Loss:",loss.item())


# -------------------------
# Save model
# -------------------------

torch.save(model.state_dict(),"snn_model.pth")

print("Model saved.")

with torch.no_grad():

    preds = model(X)

    predicted = (preds > 0.5).float()

    accuracy = (predicted == y).float().mean()

    print("Accuracy:",accuracy.item())