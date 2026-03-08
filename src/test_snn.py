import torch
import torch.nn as nn
import csv
import numpy as np
from feature_extractor import extract_features

# -------------------------
# LIF neuron
# -------------------------

class LIFNeuron(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.threshold = 1.0
        self.decay = 0.9

    def forward(self, x):

        mem = torch.zeros_like(x)

        mem = self.decay * mem + x

        spikes = (mem > self.threshold).float()

        mem = mem * (1 - spikes)

        return spikes


# -------------------------
# SNN model (same as training)
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
# Load trained model
# -------------------------

model = SNN()
model.load_state_dict(torch.load("snn_model.pth"))
model.eval()

print("Trained model loaded ✅")


# -------------------------
# Test file
# -------------------------

TEST_FILE = r"C:\Users\USER\hand-movement-snn\testfile\user20_led.aedat"
LABEL_FILE = r"C:\Users\USER\hand-movement-snn\testfile\user20_led_labels.csv"

print("\nGesture Predictions")
print("-------------------")


with open(LABEL_FILE, "r") as f:

    reader = csv.reader(f)
    next(reader)

    rows = list(reader)

    print("Total gestures found:", len(rows))

    for row in rows:

        gesture = int(row[0])
        start_time = int(row[1])
        end_time = int(row[2])

        features = extract_features(
        TEST_FILE,
        start_time,
        end_time
             )

        if features is None:
            print("Skipping gesture", gesture, "- no events found")
            continue
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():

            output = model(x)

            prediction = (output > 0.5).float().item()

        if prediction == 1:
            result = "Hand"
        else:
            result = "No Hand"

        print("Gesture", gesture, ":", result)